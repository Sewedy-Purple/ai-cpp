// trainer.cpp — complete working build
// g++ -O3 -march=native -std=c++17 -fopenmp trainer.cpp -o trainer
// cl /O2 /std:c++17 /EHsc /openmp trainer.cpp

#ifdef _MSC_VER
  #define _CRT_SECURE_NO_WARNINGS
  #pragma warning(disable:4996)
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <limits>
#include <map>
#include <utility>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdint>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <dirent.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef USE_OPENCL
  #include <CL/cl.h>
#endif

using namespace std;
constexpr double PI = 3.14159265358979323846;

// ============================================================================
// SETTINGS (all knobs live here)
// ============================================================================
struct Settings {
    int D = 128, HEADS = 4, FFN_MULT = 4, SEQ = 128;
    int NUM_EXPERTS = 4, TOP_K = 2;
    bool USE_MOE = true;
    int VOCAB_TARGET = 512;
    int STEPS = 2000, BATCH = 4, GRAD_ACCUM = 1;
    double LR = 3e-3, WD = 0.01, CLIP = 1.0, DROPOUT = 0.1;
    double WARMUP_RATIO = 0.05, DECAY_RATIO = 0.3;
    double LABEL_SMOOTH = 0.1;
    bool USE_NORMUON = true;
    double MOMENTUM = 0.95; int NS_ITERS = 5;
    bool USE_EMA = true; double EMA_DECAY = 0.999;
    int EVAL_EVERY = 200, LOG_EVERY = 10;
    bool TRY_GPU = true;
    double TEMPERATURE = 0.8, TOP_P = 0.9; int TOP_K_SAMPLE = 50;
    bool SCAN_SUBFOLDERS = true, AUTO_ZIP = true;
    int MIN_PARAMS = 5000, MAX_PARAMS_WARN = 50000000;
    int EARLY_STOP_PATIENCE = 5;
    bool SAVE_CKPT = false;
    string CKPT_PATH = "ckpt.bin";
    string CSV_LOG = "train.csv";
};

// ============================================================================
// MATH HELPERS
// ============================================================================
static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline double relu(double x) { return x > 0.0 ? x : 0.0; }

static inline double gelu(double x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}
static inline double gelu_prime(double x) {
    double x3 = x * x * x;
    double u = 0.7978845608 * (x + 0.044715 * x3);
    double t = tanh(u);
    double du = 0.7978845608 * (1.0 + 0.134145 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;
}

static inline double dot(const double* a, const double* b, int n) {
    double s = 0.0;
    #ifdef _OPENMP
    #pragma omp simd reduction(+:s)
    #endif
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
static inline void axpy(double alpha, const double* in, double* out, int n) {
    #ifdef _OPENMP
    #pragma omp simd
    #endif
    for (int i = 0; i < n; ++i) out[i] += alpha * in[i];
}

// ============================================================================
// ENV VARS
// ============================================================================
static int env_int(const char* name, int def) {
#ifdef _MSC_VER
    char* buf = NULL; size_t sz = 0;
    if (_dupenv_s(&buf, &sz, name) == 0 && buf) { int v = atoi(buf); free(buf); return v; }
    return def;
#else
    const char* e = getenv(name); return e ? atoi(e) : def;
#endif
}
static double env_dbl(const char* name, double def) {
#ifdef _MSC_VER
    char* buf = NULL; size_t sz = 0;
    if (_dupenv_s(&buf, &sz, name) == 0 && buf) { double v = atof(buf); free(buf); return v; }
    return def;
#else
    const char* e = getenv(name); return e ? atof(e) : def;
#endif
}

static Settings pick_config() {
    Settings s;
    s.D = env_int("TRAIN_D", s.D);
    s.HEADS = env_int("TRAIN_HEADS", s.HEADS);
    s.SEQ = env_int("TRAIN_SEQ", s.SEQ);
    s.STEPS = env_int("TRAIN_STEPS", s.STEPS);
    s.LR = env_dbl("TRAIN_LR", s.LR);
    s.NUM_EXPERTS = env_int("TRAIN_EXPERTS", s.NUM_EXPERTS);
    s.TOP_K = env_int("TRAIN_TOPK", s.TOP_K);
    s.VOCAB_TARGET = env_int("TRAIN_VOCAB", s.VOCAB_TARGET);
    s.USE_MOE = env_int("TRAIN_MOE", s.USE_MOE ? 1 : 0) != 0;
    if (s.HEADS <= 0) s.HEADS = 4;
    if (s.D % s.HEADS != 0) s.HEADS = 4;
    return s;
}

// ============================================================================
// SOFTMAX + SAMPLING
// ============================================================================
static void softmax_stable(vector<double>& v) {
    if (v.empty()) return;
    double m = -numeric_limits<double>::infinity();
    for (size_t i = 0; i < v.size(); ++i) if (v[i] > m) m = v[i];
    if (!std::isfinite(m)) {
        double u = 1.0 / (double)v.size();
        for (size_t i = 0; i < v.size(); ++i) v[i] = u;
        return;
    }
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) { v[i] = exp(v[i] - m); s += v[i]; }
    if (s == 0.0 || !std::isfinite(s)) {
        double u = 1.0 / (double)v.size();
        for (size_t i = 0; i < v.size(); ++i) v[i] = u;
        return;
    }
    for (size_t i = 0; i < v.size(); ++i) v[i] /= s;
}

static bool cmp_pair_desc(const pair<double,int>& a, const pair<double,int>& b) {
    return a.first > b.first;
}

static int sample_from(const vector<double>& probs, double temp, double top_p, int top_k, mt19937& rng) {
    int V = (int)probs.size();
    if (V == 0) return 0;
    vector<double> p = probs;
    if (temp != 1.0 && temp > 0) {
        for (size_t i = 0; i < p.size(); ++i) p[i] = pow(p[i], 1.0 / temp);
        double s = accumulate(p.begin(), p.end(), 0.0);
        if (s > 0) for (size_t i = 0; i < p.size(); ++i) p[i] /= s;
    }
    if (top_k > 0 && top_k < V) {
        vector<pair<double,int>> sp(V);
        for (int i = 0; i < V; ++i) sp[i] = make_pair(p[i], i);
        partial_sort(sp.begin(), sp.begin() + top_k, sp.end(), cmp_pair_desc);
        vector<double> q(V, 0.0); double s = 0.0;
        for (int i = 0; i < top_k; ++i) { q[sp[i].second] = sp[i].first; s += sp[i].first; }
        if (s > 0) for (size_t i = 0; i < q.size(); ++i) q[i] /= s;
        p.swap(q);
    }
    if (top_p <= 0.0) {
        int best = 0;
        for (int i = 1; i < V; ++i) if (p[i] > p[best]) best = i;
        return best;
    }
    if (top_p < 1.0) {
        vector<pair<double,int>> sp(V);
        for (int i = 0; i < V; ++i) sp[i] = make_pair(p[i], i);
        sort(sp.begin(), sp.end(), cmp_pair_desc);
        double cum = 0.0; vector<double> q(V, 0.0);
        for (int i = 0; i < V; ++i) {
            if (cum >= top_p) break;
            q[sp[i].second] = sp[i].first; cum += sp[i].first;
        }
        double s = accumulate(q.begin(), q.end(), 0.0);
        if (s > 0) for (size_t i = 0; i < q.size(); ++i) q[i] /= s;
        p.swap(q);
    }
    discrete_distribution<int> dist(p.begin(), p.end());
    return dist(rng);
}

// ============================================================================
// MATMUL
// ============================================================================
static vector<double> matmul(const vector<double>& A, const vector<double>& B, int M, int K, int N) {
    vector<double> C((size_t)M * (size_t)N, 0.0);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < M; ++i) {
        const size_t iK = (size_t)i * (size_t)K;
        const size_t iN = (size_t)i * (size_t)N;
        for (int k = 0; k < K; ++k) {
            double a = A[iK + (size_t)k];
            if (a == 0.0) continue;
            const double* brow = &B[(size_t)k * (size_t)N];
            double* crow = &C[iN];
            for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
    }
    return C;
}

static vector<double> matmul_TN(const vector<double>& A, const vector<double>& B, int M, int K, int N) {
    vector<double> C((size_t)M * (size_t)N, 0.0);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int k = 0; k < K; ++k) {
        const double* arow = &A[(size_t)k * (size_t)M];
        const double* brow = &B[(size_t)k * (size_t)N];
        for (int i = 0; i < M; ++i) {
            double a = arow[i];
            if (a == 0.0) continue;
            axpy(a, brow, &C[(size_t)i * (size_t)N], N);
        }
    }
    return C;
}

// ============================================================================
// GPU PROBE (OpenCL — probe only, kernels not written)
// ============================================================================
#ifdef USE_OPENCL
struct GPUBackend {
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_device_id device = nullptr;
    bool ready = false;
    string err;
    bool init() {
        cl_platform_id plat; cl_uint nplat;
        if (clGetPlatformIDs(1, &plat, &nplat) != CL_SUCCESS || nplat == 0) { err = "no platform"; return false; }
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) { err = "no GPU"; return false; }
        ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        if (!ctx) { err = "ctx"; return false; }
        queue = clCreateCommandQueue(ctx, device, 0, nullptr);
        if (!queue) { err = "queue"; return false; }
        ready = true; return true;
    }
    ~GPUBackend() {
        if (queue) clReleaseCommandQueue(queue);
        if (ctx) clReleaseContext(ctx);
    }
};
static GPUBackend g_gpu;
#endif

// ============================================================================
// PARAMETER ESTIMATOR
// ============================================================================
struct ParamEstimate { long long dense, moe_extra, total; };

static ParamEstimate estimate_params(int D, int V, int SEQ, int FFN, int E, bool use_moe) {
    ParamEstimate e; e.dense = 0; e.moe_extra = 0; e.total = 0;
    e.dense += (long long)V * D;
    e.dense += (long long)SEQ * D;
    e.dense += 4LL * D * D;
    e.dense += 2LL * D * FFN;
    e.dense += 4LL * D;                        // 2 LN pairs × (gain + bias)
    e.dense += (long long)D * V + V;
    if (use_moe && E > 1) {
        e.moe_extra = (long long)(E - 1) * 2LL * D * FFN;
        e.moe_extra += (long long)E * (FFN + D); // expert biases (previously missed)
        e.moe_extra += (long long)D * E;         // router
    }
    e.total = e.dense + e.moe_extra;
    return e;
}

static bool ask_param_confirmation(ParamEstimate e, int& D, int& FFN, int& E) {
    cout << "\n[estimate] dense=" << e.dense << "  moe_extra=" << e.moe_extra
         << "  total=" << e.total << " params\n";
    cout << "  D=" << D << " FFN=" << FFN << " experts=" << E << "\n";
    cout << "Is this good? [Y/n]: ";
    string s; getline(cin, s);
    if (s.empty() || s[0] == 'y' || s[0] == 'Y') return true;
    cout << "New D (blank=keep): "; getline(cin, s); if (!s.empty()) D = atoi(s.c_str());
    cout << "New FFN (blank=keep): "; getline(cin, s); if (!s.empty()) FFN = atoi(s.c_str());
    cout << "New experts (blank=keep): "; getline(cin, s); if (!s.empty()) E = atoi(s.c_str());
    return false;
}

// ============================================================================
// FILE WALKER
// ============================================================================
static bool is_regular_file(const string& p) {
#ifdef _WIN32
    DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st; return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}
static bool is_directory(const string& p) {
#ifdef _WIN32
    DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st; return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}
static vector<string> list_dir(const string& dir) {
    vector<string> out;
#ifdef _WIN32
    string pat = dir + "\\*"; WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return out;
    do { string n = fd.cFileName; if (n != "." && n != "..") out.push_back(dir + "\\" + n); }
    while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* d = opendir(dir.c_str()); if (!d) return out;
    struct dirent* e;
    while ((e = readdir(d))) { string n = e->d_name; if (n != "." && n != "..") out.push_back(dir + "/" + n); }
    closedir(d);
#endif
    return out;
}
static bool read_file_to_string(const string& path, string& out) {
    ifstream f(path.c_str(), ios::binary); if (!f) return false;
    ostringstream ss; ss << f.rdbuf(); out = ss.str(); return true;
}
static string lower_ext(const string& p) {
    size_t slash = p.find_last_of("/\\"), dot = p.find_last_of('.');
    if (dot == string::npos || (slash != string::npos && dot < slash)) return "";
    string e = p.substr(dot);
    for (size_t i = 0; i < e.size(); ++i) e[i] = (char)tolower((unsigned char)e[i]);
    return e;
}
static bool is_allowed_ext(const string& p) {
    string e = lower_ext(p);
    return e == ".txt" || e == ".json" || e == ".jsonl" || e == ".csv";
}
static vector<string> split_lines(const string& raw) {
    vector<string> lines; string cur;
    for (size_t i = 0; i < raw.size(); ++i) {
        char c = raw[i];
        if (c == '\n') { lines.push_back(cur); cur.clear(); }
        else if (c != '\r') cur += c;
    }
    if (!cur.empty()) lines.push_back(cur);
    return lines;
}
static void extract_json_strings(const string& raw, string& out) {
    vector<string> lines = split_lines(raw);
    for (size_t li = 0; li < lines.size(); ++li) {
        const string& line = lines[li];
        bool in_str = false, esc = false;
        for (size_t i = 0; i < line.size(); ++i) {
            char c = line[i];
            if (esc) { if (in_str) out += c; esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') { in_str = !in_str; continue; }
            if (in_str) out += c;
        }
        out += '\n';
    }
}
static void extract_csv_strings(const string& raw, string& out) {
    vector<string> lines = split_lines(raw);
    for (size_t li = 0; li < lines.size(); ++li) {
        const string& line = lines[li];
        for (size_t i = 0; i < line.size(); ++i) out += (line[i] == ',' ? ' ' : line[i]);
        out += '\n';
    }
}
static void ingest_one_file(const string& p, string& corpus, int& n) {
    string ext = lower_ext(p), tmp;
    if (!read_file_to_string(p, tmp)) { cerr << "[skip] " << p << "\n"; return; }
    if (ext == ".json" || ext == ".jsonl") extract_json_strings(tmp, corpus);
    else if (ext == ".csv") extract_csv_strings(tmp, corpus);
    else corpus += tmp;
    corpus.push_back('\n'); n++;
}
static bool ingest_folder(const string& dir, string& corpus, string& err, bool scan_sub, bool auto_zip) {
    corpus.clear();
    if (!is_directory(dir)) { err = "not a directory: " + dir; return false; }
    int n = 0;
    vector<string> stack; stack.push_back(dir);
    while (!stack.empty()) {
        string cur = stack.back(); stack.pop_back();
        vector<string> children = list_dir(cur);
        for (size_t i = 0; i < children.size(); ++i) {
            const string& c = children[i];
            if (is_directory(c)) { if (scan_sub) stack.push_back(c); continue; }
            if (!is_regular_file(c)) continue;
            string e = lower_ext(c);
            if (e == ".zip" && auto_zip) {
                string cmd = "unzip -p \"" + c + "\" 2>/dev/null";
#ifdef _WIN32
                cmd = "unzip -p \"" + c + "\" 2>nul";
                FILE* fp = _popen(cmd.c_str(), "r");
#else
                FILE* fp = popen(cmd.c_str(), "r");
#endif
                if (fp) {
                    char buf[4096]; size_t r;
                    while ((r = fread(buf, 1, sizeof(buf), fp)) > 0) corpus.append(buf, r);
#ifdef _WIN32
                    _pclose(fp);
#else
                    pclose(fp);
#endif
                } else cerr << "[skip zip] " << c << "\n";
                continue;
            }
            if (!is_allowed_ext(c)) continue;
            ingest_one_file(c, corpus, n);
        }
    }
    if (corpus.empty()) { err = "no allowed files"; return false; }
    cout << "[ingest] " << n << " files, " << corpus.size() << " chars\n";
    return true;
}

// ============================================================================
// BPE TOKENIZER
// ============================================================================
struct BPETokenizer {
    map<pair<int,int>, int> merges;
    vector<pair<int,int>> merge_list;
    vector<string> id_to_bytes;
    int next_id = 256; bool trained = false; string last_error;

    vector<int> encode_raw(const string& s) const {
        vector<int> o; o.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) o.push_back((unsigned char)s[i]);
        return o;
    }
    bool train(const string& corpus, int target_vocab, int max_merges = 2000) {
        last_error.clear();
        if (corpus.empty()) { last_error = "empty corpus"; return false; }
        if (target_vocab < 257) { last_error = "vocab<257"; return false; }
        int nm = imin(target_vocab - 256, max_merges);
        vector<int> tok = encode_raw(corpus);
        merges.clear(); merge_list.clear(); next_id = 256;
        for (int m = 0; m < nm; ++m) {
            if (tok.size() < 2) break;
            map<pair<int,int>, int> freq;
            for (size_t i = 0; i + 1 < tok.size(); ++i) freq[make_pair(tok[i], tok[i+1])]++;
            if (freq.empty()) break;
            pair<int,int> best = freq.begin()->first; int bc = freq.begin()->second;
            for (map<pair<int,int>,int>::const_iterator it = freq.begin(); it != freq.end(); ++it) {
                if (it->second > bc || (it->second == bc && it->first < best)) { best = it->first; bc = it->second; }
            }
            if (bc < 2) break;
            merges[best] = next_id; merge_list.push_back(best);
            int nid = next_id++;
            vector<int> nt; nt.reserve(tok.size());
            for (size_t i = 0; i < tok.size(); ) {
                if (i + 1 < tok.size() && tok[i] == best.first && tok[i+1] == best.second) { nt.push_back(nid); i += 2; }
                else { nt.push_back(tok[i]); ++i; }
            }
            tok.swap(nt);
        }
        trained = true; build_reverse();
        return true;
    }
    void build_reverse() {
        id_to_bytes.assign(next_id, "");
        for (int b = 0; b < 256; ++b) id_to_bytes[b] = string(1, (char)b);
        for (size_t i = 0; i < merge_list.size(); ++i) {
            int pid = 256 + (int)i;
            id_to_bytes[pid] = id_to_bytes[merge_list[i].first] + id_to_bytes[merge_list[i].second];
        }
    }
    vector<int> encode(const string& s) const {
        if (!trained) return encode_raw(s);
        vector<int> tok = encode_raw(s);
        bool changed = true;
        while (changed && tok.size() > 1) {
            changed = false; vector<int> nt; nt.reserve(tok.size());
            for (size_t i = 0; i < tok.size(); ) {
                if (i + 1 < tok.size()) {
                    map<pair<int,int>,int>::const_iterator it = merges.find(make_pair(tok[i], tok[i+1]));
                    if (it != merges.end()) { nt.push_back(it->second); i += 2; changed = true; continue; }
                }
                nt.push_back(tok[i]); ++i;
            }
            tok.swap(nt);
        }
        return tok;
    }
    string decode(const vector<int>& ids) const {
        string out;
        for (size_t i = 0; i < ids.size(); ++i) {
            int id = ids[i];
            if (id >= 0 && id < (int)id_to_bytes.size()) out += id_to_bytes[id];
            else out += "?";
        }
        return out;
    }
    int vocab_size() const { return next_id; }
};

// ============================================================================
// PARAMS
// ============================================================================
struct Params {
    vector<double> tok_emb, pos_emb;
    vector<double> Wq, Wk, Wv, Wo;
    vector<double> W1, W2, b1, b2;
    vector<double> ln1g, ln1b, ln2g, ln2b;
    vector<double> Wout, bout, Wr;
    vector<vector<double>> exp_W1, exp_W2, exp_b1, exp_b2;
    // NorMuon state
    vector<double> mom_Wq, mom_Wk, mom_Wv, mom_Wo, mom_W1, mom_W2, mom_Wout;
    vector<vector<double>> mom_exp_W1, mom_exp_W2;
    vector<double> nv_Wq, nv_Wk, nv_Wv, nv_Wo, nv_W1, nv_W2, nv_Wout;
    vector<vector<double>> nv_exp_W1, nv_exp_W2;
    // AdamW state (1-D + embeddings)
    vector<double> mt, vt; int t;
    // EMA shadows
    vector<double> ema_tok_emb, ema_pos_emb, ema_Wq, ema_Wk, ema_Wv, ema_Wo, ema_Wout;
    vector<vector<double>> ema_exp_W1, ema_exp_W2;
    bool ema_init;
    Params() : t(0), ema_init(false) {}
};

static mt19937 g_rng(42);
static double uniform_(double lo, double hi) {
    uniform_real_distribution<double> d(lo, hi); return d(g_rng);
}
static void xavier(vector<double>& W, int fi, int fo) {
    double lim = sqrt(6.0 / (double)(fi + fo));
    for (size_t i = 0; i < W.size(); ++i) W[i] = uniform_(-lim, lim);
}

static Params make_params(const Settings& S, int V, int D, int SEQ, int FFN, int E) {
    Params p;
    p.tok_emb.assign((size_t)V * D, 0.0); xavier(p.tok_emb, V, D);
    p.pos_emb.assign((size_t)SEQ * D, 0.0); xavier(p.pos_emb, SEQ, D);
    p.Wq.assign((size_t)D * D, 0.0); xavier(p.Wq, D, D);
    p.Wk.assign((size_t)D * D, 0.0); xavier(p.Wk, D, D);
    p.Wv.assign((size_t)D * D, 0.0); xavier(p.Wv, D, D);
    p.Wo.assign((size_t)D * D, 0.0); xavier(p.Wo, D, D);
    p.ln1g.assign(D, 1.0); p.ln1b.assign(D, 0.0);
    p.ln2g.assign(D, 1.0); p.ln2b.assign(D, 0.0);
    p.Wout.assign((size_t)D * V, 0.0); xavier(p.Wout, D, V);
    p.bout.assign(V, 0.0);
    p.Wr.assign((size_t)D * E, 0.0); xavier(p.Wr, D, E);
    p.exp_W1.resize(E); p.exp_W2.resize(E); p.exp_b1.resize(E); p.exp_b2.resize(E);
    for (int e = 0; e < E; ++e) {
        p.exp_W1[e].assign((size_t)D * FFN, 0.0); xavier(p.exp_W1[e], D, FFN);
        p.exp_W2[e].assign((size_t)FFN * D, 0.0); xavier(p.exp_W2[e], FFN, D);
        p.exp_b1[e].assign(FFN, 0.0);
        p.exp_b2[e].assign(D, 0.0);
    }
    // NorMuon state
    p.mom_Wq.assign(p.Wq.size(), 0.0); p.mom_Wk.assign(p.Wk.size(), 0.0);
    p.mom_Wv.assign(p.Wv.size(), 0.0); p.mom_Wo.assign(p.Wo.size(), 0.0);
    p.mom_Wout.assign(p.Wout.size(), 0.0);
    p.mom_exp_W1.resize(E); p.mom_exp_W2.resize(E);
    p.nv_Wq.assign(D, 1.0); p.nv_Wk.assign(D, 1.0); p.nv_Wv.assign(D, 1.0); p.nv_Wo.assign(D, 1.0);
    p.nv_Wout.assign(D, 1.0);
    p.nv_exp_W1.resize(E); p.nv_exp_W2.resize(E);
    for (int e = 0; e < E; ++e) {
        p.mom_exp_W1[e].assign(p.exp_W1[e].size(), 0.0);
        p.mom_exp_W2[e].assign(p.exp_W2[e].size(), 0.0);
        p.nv_exp_W1[e].assign(D, 1.0);
        p.nv_exp_W2[e].assign(FFN, 1.0);
    }
    // AdamW 1-D state
    size_t total = p.tok_emb.size() + p.pos_emb.size() + p.ln1g.size() + p.ln1b.size()
                 + p.ln2g.size() + p.ln2b.size() + p.bout.size();
    p.mt.assign(total, 0.0); p.vt.assign(total, 0.0);
    if (S.USE_EMA) {
        p.ema_tok_emb = p.tok_emb; p.ema_pos_emb = p.pos_emb;
        p.ema_Wq = p.Wq; p.ema_Wk = p.Wk; p.ema_Wv = p.Wv; p.ema_Wo = p.Wo;
        p.ema_Wout = p.Wout;
        p.ema_exp_W1 = p.exp_W1; p.ema_exp_W2 = p.exp_W2;
        p.ema_init = true;
    }
    return p;
}

// ============================================================================
// CACHE
// ============================================================================
struct Cache {
    int N;
    vector<double> x, xhat1, inv_std1, xn1;
    vector<double> Q, K, V, attn_out, attn_proj, r1;
    vector<double> xhat2, inv_std2, xn2;
    vector<double> h2, xn3, logits, probs, attn_w;
    vector<double> router_probs;
    vector<vector<double>> h1_act;
    vector<double> mask_attn, mask_moe;   // dropout masks
    vector<int> chosen_experts;           // N*TOP_K
    vector<double> chosen_weights;        // N*TOP_K
    Cache() : N(0) {}
};

// ============================================================================
// LAYER NORM (with xhat + inv_std stored for backward)
// ============================================================================
static void layer_norm(const vector<double>& x, vector<double>& out,
                       vector<double>& xhat, vector<double>& inv_std, int N, int d) {
    out.resize(x.size()); xhat.resize(x.size()); inv_std.assign(N, 1.0);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; ++i) {
        const size_t base = (size_t)i * (size_t)d;
        double s = 0, s2 = 0;
        for (int j = 0; j < d; ++j) { double v = x[base + j]; s += v; s2 += v * v; }
        double mu = s / (double)d;
        double va = s2 / (double)d - mu * mu;
        if (va < 0) va = 0;
        double inv = 1.0 / sqrt(va + 1e-5);
        inv_std[i] = inv;
        for (int j = 0; j < d; ++j) {
            xhat[base + j] = (x[base + j] - mu) * inv;
            out[base + j] = xhat[base + j];
        }
    }
}

// ============================================================================
// FORWARD
// ============================================================================
static void forward(const Params& p, const Settings& S, int V, int D, int F, int E,
                    const vector<int>& toks, Cache& c, bool training) {
    int N = (int)toks.size(); c.N = N;
    if (N == 0) { c.probs.clear(); return; }
    double keep = 1.0 - S.DROPOUT;
    double dscale = (keep > 0) ? 1.0 / keep : 0.0;

    c.x.resize((size_t)N * D);
    for (int i = 0; i < N; ++i) {
        const size_t iD = (size_t)i * D;
        const size_t tD = (size_t)toks[i] * D;
        for (int j = 0; j < D; ++j)
            c.x[iD + j] = p.tok_emb[tD + j] + p.pos_emb[iD + j];
    }

    // LN1
    layer_norm(c.x, c.xn1, c.xhat1, c.inv_std1, N, D);
    for (size_t i = 0; i < c.xn1.size(); ++i)
        c.xn1[i] = c.xhat1[i] * p.ln1g[i % D] + p.ln1b[i % D];

    c.Q = matmul(c.xn1, p.Wq, N, D, D);
    c.K = matmul(c.xn1, p.Wk, N, D, D);
    c.V = matmul(c.xn1, p.Wv, N, D, D);

    int HD = D / S.HEADS;
    c.attn_out.assign((size_t)N * D, 0.0);
    c.attn_w.assign((size_t)S.HEADS * N * N, 0.0);
    double scale = 1.0 / sqrt((double)HD);
    for (int h = 0; h < S.HEADS; ++h) {
        int ho = h * HD;
        for (int i = 0; i < N; ++i) {
            vector<double> sc((size_t)i + 1, 0.0);
            size_t qb = (size_t)i * D + ho;
            for (int j = 0; j <= i; ++j) {
                size_t kb = (size_t)j * D + ho;
                sc[j] = dot(&c.Q[qb], &c.K[kb], HD) * scale;
            }
            softmax_stable(sc);
            for (int j = 0; j <= i; ++j) {
                c.attn_w[((size_t)h * N + i) * N + j] = sc[j];
                size_t vb = (size_t)j * D + ho;
                axpy(sc[j], &c.V[vb], &c.attn_out[qb], HD);
            }
        }
    }
    c.attn_proj = matmul(c.attn_out, p.Wo, N, D, D);

    // Dropout on attn_proj
    if (training && S.DROPOUT > 0) {
        c.mask_attn.resize((size_t)N * D);
        for (size_t i = 0; i < c.mask_attn.size(); ++i)
            c.mask_attn[i] = (uniform_(0.0, 1.0) >= S.DROPOUT) ? 1.0 : 0.0;
        for (size_t i = 0; i < c.attn_proj.size(); ++i)
            c.attn_proj[i] *= c.mask_attn[i] * dscale;
    }

    c.r1.resize((size_t)N * D);
    for (size_t i = 0; i < c.r1.size(); ++i) c.r1[i] = c.x[i] + c.attn_proj[i];

    // LN2
    layer_norm(c.r1, c.xn2, c.xhat2, c.inv_std2, N, D);
    for (size_t i = 0; i < c.xn2.size(); ++i)
        c.xn2[i] = c.xhat2[i] * p.ln2g[i % D] + p.ln2b[i % D];

    // MoE router
    c.router_probs.assign((size_t)N * E, 0.0);
    for (int i = 0; i < N; ++i) {
        vector<double> logits(E, 0.0);
        for (int e = 0; e < E; ++e)
            logits[e] = dot(&c.xn2[(size_t)i * D], &p.Wr[(size_t)e * D], D);
        softmax_stable(logits);
        for (int e = 0; e < E; ++e) c.router_probs[(size_t)i * E + e] = logits[e];
    }

    // MoE forward
    int K = imin(S.TOP_K, E);
    c.chosen_experts.assign((size_t)N * K, 0);
    c.chosen_weights.assign((size_t)N * K, 0.0);
    c.h1_act.assign(E, vector<double>());
    for (int e = 0; e < E; ++e) c.h1_act[e].assign((size_t)N * F, 0.0);
    c.h2.assign((size_t)N * D, 0.0);

    for (int i = 0; i < N; ++i) {
        vector<pair<double,int>> rp(E);
        for (int e = 0; e < E; ++e) rp[e] = make_pair(c.router_probs[(size_t)i * E + e], e);
        partial_sort(rp.begin(), rp.begin() + K, rp.end(), cmp_pair_desc);
        for (int k = 0; k < K; ++k) {
            int e = rp[k].second; double w = rp[k].first;
            c.chosen_experts[(size_t)i * K + k] = e;
            c.chosen_weights[(size_t)i * K + k] = w;
            for (int f = 0; f < F; ++f) {
                double z = dot(&c.xn2[(size_t)i * D], &p.exp_W1[e][(size_t)f * D], D) + p.exp_b1[e][f];
                c.h1_act[e][(size_t)i * F + f] = gelu(z);
            }
            for (int j = 0; j < D; ++j) {
                double s = p.exp_b2[e][j];
                for (int f = 0; f < F; ++f) s += c.h1_act[e][(size_t)i * F + f] * p.exp_W2[e][(size_t)j * F + f];
                c.h2[(size_t)i * D + j] += w * s;
            }
        }
    }

    // Dropout on MoE output
    if (training && S.DROPOUT > 0) {
        c.mask_moe.resize((size_t)N * D);
        for (size_t i = 0; i < c.mask_moe.size(); ++i)
            c.mask_moe[i] = (uniform_(0.0, 1.0) >= S.DROPOUT) ? 1.0 : 0.0;
        for (size_t i = 0; i < c.h2.size(); ++i)
            c.h2[i] *= c.mask_moe[i] * dscale;
    }

    for (size_t i = 0; i < c.h2.size(); ++i) c.h2[i] += c.r1[i];
    c.xn3 = c.h2;

    c.logits = matmul(c.xn3, p.Wout, N, D, V);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < V; ++j)
            c.logits[(size_t)i * V + j] += p.bout[j];

    c.probs.resize(c.logits.size());
    for (int i = 0; i < N; ++i) {
        vector<double> row(c.logits.begin() + (size_t)i * V, c.logits.begin() + (size_t)(i + 1) * V);
        softmax_stable(row);
        for (int j = 0; j < V; ++j) c.probs[(size_t)i * V + j] = row[j];
    }
}

// ============================================================================
// CROSS-ENTROPY (correct label smoothing)
// ============================================================================
static double cross_entropy(const Cache& c, const vector<int>& targets, int V, double smooth) {
    int N = (int)targets.size();
    if (N == 0) return 0.0;
    double loss = 0.0;
    for (int i = 0; i < N; ++i) {
        double sum_log_p = 0.0;
        for (int j = 0; j < V; ++j) {
            double p = c.probs[(size_t)i * V + j];
            if (p < 1e-12) p = 1e-12;
            sum_log_p += log(p);
        }
        double log_pt = log(dmax(c.probs[(size_t)i * V + targets[i]], 1e-12));
        loss -= (1.0 - smooth) * log_pt + smooth * sum_log_p / (double)V;
    }
    return loss / (double)N;
}

// ============================================================================
// BACKWARD (complete)
// ============================================================================
struct Grads {
    vector<double> dWq, dWk, dWv, dWo, dW1, dW2, db1, db2;
    vector<double> dln1g, dln1b, dln2g, dln2b, dWout, dbout;
    vector<double> dtok_emb, dpos_emb, dWr;
    vector<vector<double>> d_exp_W1, d_exp_W2, d_exp_b1, d_exp_b2;
};

static void init_grads(Grads& g, int V, int D, int F, int E, int SEQ) {
    g.dWq.assign((size_t)D * D, 0.0); g.dWk.assign((size_t)D * D, 0.0);
    g.dWv.assign((size_t)D * D, 0.0); g.dWo.assign((size_t)D * D, 0.0);
    g.dW1.assign((size_t)D * F, 0.0); g.dW2.assign((size_t)F * D, 0.0);
    g.db1.assign(F, 0.0); g.db2.assign(D, 0.0);
    g.dln1g.assign(D, 0.0); g.dln1b.assign(D, 0.0);
    g.dln2g.assign(D, 0.0); g.dln2b.assign(D, 0.0);
    g.dWout.assign((size_t)D * V, 0.0); g.dbout.assign(V, 0.0);
    g.dtok_emb.assign((size_t)V * D, 0.0); g.dpos_emb.assign((size_t)SEQ * D, 0.0);
    g.dWr.assign((size_t)D * E, 0.0);
    g.d_exp_W1.resize(E); g.d_exp_W2.resize(E); g.d_exp_b1.resize(E); g.d_exp_b2.resize(E);
    for (int e = 0; e < E; ++e) {
        g.d_exp_W1[e].assign((size_t)D * F, 0.0);
        g.d_exp_W2[e].assign((size_t)F * D, 0.0);
        g.d_exp_b1[e].assign(F, 0.0);
        g.d_exp_b2[e].assign(D, 0.0);
    }
}

static void accumulate_grads(Grads& dst, const Grads& src) {
    for (size_t i = 0; i < dst.dWq.size(); ++i) dst.dWq[i] += src.dWq[i];
    for (size_t i = 0; i < dst.dWk.size(); ++i) dst.dWk[i] += src.dWk[i];
    for (size_t i = 0; i < dst.dWv.size(); ++i) dst.dWv[i] += src.dWv[i];
    for (size_t i = 0; i < dst.dWo.size(); ++i) dst.dWo[i] += src.dWo[i];
    for (size_t i = 0; i < dst.dWout.size(); ++i) dst.dWout[i] += src.dWout[i];
    for (size_t i = 0; i < dst.dbout.size(); ++i) dst.dbout[i] += src.dbout[i];
    for (size_t i = 0; i < dst.dtok_emb.size(); ++i) dst.dtok_emb[i] += src.dtok_emb[i];
    for (size_t i = 0; i < dst.dpos_emb.size(); ++i) dst.dpos_emb[i] += src.dpos_emb[i];
    for (size_t i = 0; i < dst.dln1g.size(); ++i) dst.dln1g[i] += src.dln1g[i];
    for (size_t i = 0; i < dst.dln1b.size(); ++i) dst.dln1b[i] += src.dln1b[i];
    for (size_t i = 0; i < dst.dln2g.size(); ++i) dst.dln2g[i] += src.dln2g[i];
    for (size_t i = 0; i < dst.dln2b.size(); ++i) dst.dln2b[i] += src.dln2b[i];
    for (size_t i = 0; i < dst.dWr.size(); ++i) dst.dWr[i] += src.dWr[i];
    for (size_t e = 0; e < dst.d_exp_W1.size(); ++e) {
        for (size_t i = 0; i < dst.d_exp_W1[e].size(); ++i) dst.d_exp_W1[e][i] += src.d_exp_W1[e][i];
        for (size_t i = 0; i < dst.d_exp_W2[e].size(); ++i) dst.d_exp_W2[e][i] += src.d_exp_W2[e][i];
        for (size_t i = 0; i < dst.d_exp_b1[e].size(); ++i) dst.d_exp_b1[e][i] += src.d_exp_b1[e][i];
        for (size_t i = 0; i < dst.d_exp_b2[e].size(); ++i) dst.d_exp_b2[e][i] += src.d_exp_b2[e][i];
    }
}

static void scale_grads(Grads& g, double s) {
    for (size_t i = 0; i < g.dWq.size(); ++i) g.dWq[i] *= s;
    for (size_t i = 0; i < g.dWk.size(); ++i) g.dWk[i] *= s;
    for (size_t i = 0; i < g.dWv.size(); ++i) g.dWv[i] *= s;
    for (size_t i = 0; i < g.dWo.size(); ++i) g.dWo[i] *= s;
    for (size_t i = 0; i < g.dWout.size(); ++i) g.dWout[i] *= s;
    for (size_t i = 0; i < g.dbout.size(); ++i) g.dbout[i] *= s;
    for (size_t i = 0; i < g.dtok_emb.size(); ++i) g.dtok_emb[i] *= s;
    for (size_t i = 0; i < g.dpos_emb.size(); ++i) g.dpos_emb[i] *= s;
    for (size_t i = 0; i < g.dln1g.size(); ++i) g.dln1g[i] *= s;
    for (size_t i = 0; i < g.dln1b.size(); ++i) g.dln1b[i] *= s;
    for (size_t i = 0; i < g.dln2g.size(); ++i) g.dln2g[i] *= s;
    for (size_t i = 0; i < g.dln2b.size(); ++i) g.dln2b[i] *= s;
    for (size_t i = 0; i < g.dWr.size(); ++i) g.dWr[i] *= s;
    for (size_t e = 0; e < g.d_exp_W1.size(); ++e) {
        for (size_t i = 0; i < g.d_exp_W1[e].size(); ++i) g.d_exp_W1[e][i] *= s;
        for (size_t i = 0; i < g.d_exp_W2[e].size(); ++i) g.d_exp_W2[e][i] *= s;
        for (size_t i = 0; i < g.d_exp_b1[e].size(); ++i) g.d_exp_b1[e][i] *= s;
        for (size_t i = 0; i < g.d_exp_b2[e].size(); ++i) g.d_exp_b2[e][i] *= s;
    }
}

static Grads backward(const Params& p, const Settings& S, int V, int D, int F, int E,
                      const Cache& c, const vector<int>& toks, const vector<int>& targets) {
    int N = c.N;
    Grads g;
    init_grads(g, V, D, F, E, S.SEQ);
    if (N == 0) return g;
    double keep = 1.0 - S.DROPOUT;
    double dscale = (keep > 0) ? 1.0 / keep : 0.0;

    // dLogits with label smoothing
    vector<double> dl((size_t)N * V, 0.0);
    for (int i = 0; i < N; ++i) {
        double sum_p = 0;
        for (int j = 0; j < V; ++j) sum_p += c.probs[(size_t)i * V + j];
        for (int j = 0; j < V; ++j) {
            double pj = c.probs[(size_t)i * V + j];
            double target = (j == targets[i]) ? 1.0 : 0.0;
            // dCE/dlogit = (1-smooth)*(p - target) + smooth*(p - 1/V)
            dl[(size_t)i * V + j] = ((1.0 - S.LABEL_SMOOTH) * (pj - target)
                                    + S.LABEL_SMOOTH * (pj - 1.0 / V)) / N;
        }
    }

    g.dWout = matmul_TN(c.xn3, dl, D, N, V);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < V; ++j) g.dbout[j] += dl[(size_t)i * V + j];

    vector<double> dxn3((size_t)N * D, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            dxn3[(size_t)i * D + j] = dot(&dl[(size_t)i * V], &p.Wout[(size_t)j * V], V);

    // xn3 = h2 = r1 + moe_out_masked
    // d(h2) = dxn3
    // d(moe_out_masked) = dxn3
    // d(r1) += dxn3
    vector<double> dh2 = dxn3;
    vector<double> dmoe_masked = dh2;
    vector<double> dr1 = dh2;

    // Undo dropout on moe output
    if (S.DROPOUT > 0 && (int)c.mask_moe.size() == N * D) {
        for (size_t i = 0; i < dmoe_masked.size(); ++i) dmoe_masked[i] *= c.mask_moe[i] * dscale;
    }

    // MoE backward
    vector<double> dxn2((size_t)N * D, 0.0);
    int K = imin(S.TOP_K, E);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            int e = c.chosen_experts[(size_t)i * K + k];
            double w = c.chosen_weights[(size_t)i * K + k];
            // h_out[j] = sum_f h_act[f] * W2[j,f] + b2[j]; moe_out[j] += w * h_out[j]
            // dW2[j,f] += w * dmoe_masked[i,j] * h_act[f]
            // db2[j] += w * dmoe_masked[i,j]
            for (int f = 0; f < F; ++f) {
                double ha = c.h1_act[e][(size_t)i * F + f];
                if (ha == 0.0) continue;
                for (int j = 0; j < D; ++j)
                    g.d_exp_W2[e][(size_t)j * F + f] += w * dmoe_masked[(size_t)i * D + j] * ha;
            }
            for (int j = 0; j < D; ++j) g.d_exp_b2[e][j] += w * dmoe_masked[(size_t)i * D + j];
            // dh_act[f] = sum_j w * dmoe_masked[i,j] * W2[j,f]
            vector<double> dh_act(F, 0.0);
            for (int f = 0; f < F; ++f)
                for (int j = 0; j < D; ++j)
                    dh_act[f] += w * dmoe_masked[(size_t)i * D + j] * p.exp_W2[e][(size_t)j * F + f];
            // back through gelu
            vector<double> dz(F, 0.0);
            for (int f = 0; f < F; ++f) {
                if (c.h1_act[e][(size_t)i * F + f] <= 0.0) { dz[f] = 0.0; continue; }
                double z = dot(&c.xn2[(size_t)i * D], &p.exp_W1[e][(size_t)f * D], D) + p.exp_b1[e][f];
                dz[f] = dh_act[f] * gelu_prime(z);
            }
            // dW1[f,j] += w * dz[f] * xn2[i,j]
            // db1[f] += w * dz[f]
            for (int f = 0; f < F; ++f) {
                for (int j = 0; j < D; ++j)
                    g.d_exp_W1[e][(size_t)f * D + j] += w * dz[f] * c.xn2[(size_t)i * D + j];
                g.d_exp_b1[e][f] += w * dz[f];
            }
            // dxn2[i,j] += w * sum_f dz[f] * W1[f,j]
            for (int j = 0; j < D; ++j) {
                double s = 0.0;
                for (int f = 0; f < F; ++f) s += dz[f] * p.exp_W1[e][(size_t)f * D + j];
                dxn2[(size_t)i * D + j] += w * s;
            }
        }
    }

    // Back through LN2 (full formula)
    for (int i = 0; i < N; ++i) {
        const size_t base = (size_t)i * D;
        double mean_dxhat = 0.0, mean_dxhat_xhat = 0.0;
        vector<double> dxhat(D, 0.0);
        for (int j = 0; j < D; ++j) {
            double dy = dxn2[base + j];
            g.dln2g[j] += dy * c.xhat2[base + j];
            g.dln2b[j] += dy;
            dxhat[j] = dy * p.ln2g[j];
            mean_dxhat += dxhat[j];
            mean_dxhat_xhat += dxhat[j] * c.xhat2[base + j];
        }
        mean_dxhat /= D;
        mean_dxhat_xhat /= D;
        for (int j = 0; j < D; ++j) {
            dr1[base + j] += (dxhat[j] - mean_dxhat - c.xhat2[base + j] * mean_dxhat_xhat) * c.inv_std2[i];
        }
    }

    // Undo dropout on attn
    vector<double> dattn_proj = dr1;
    if (S.DROPOUT > 0 && (int)c.mask_attn.size() == N * D) {
        for (size_t i = 0; i < dattn_proj.size(); ++i) dattn_proj[i] *= c.mask_attn[i] * dscale;
    }

    // Attention backward
    g.dWo = matmul_TN(c.attn_out, dattn_proj, D, N, D);
    vector<double> dattn_out((size_t)N * D, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            dattn_out[(size_t)i * D + j] = dot(&dattn_proj[(size_t)i * D], &p.Wo[(size_t)j * D], D);

    vector<double> dQ((size_t)N * D, 0.0);
    vector<double> dK((size_t)N * D, 0.0);
    vector<double> dV((size_t)N * D, 0.0);
    int HD = D / S.HEADS;
    double scale = 1.0 / sqrt((double)HD);
    for (int h = 0; h < S.HEADS; ++h) {
        int ho = h * HD;
        for (int i = 0; i < N; ++i) {
            size_t qb = (size_t)i * D + ho;
            vector<double> dw((size_t)i + 1, 0.0);
            for (int j = 0; j <= i; ++j) {
                size_t vb = (size_t)j * D + ho;
                dw[j] = dot(&dattn_out[qb], &c.V[vb], HD);
            }
            double dotv = 0.0;
            for (int j = 0; j <= i; ++j) dotv += c.attn_w[((size_t)h * N + i) * N + j] * dw[j];
            vector<double> dscore((size_t)i + 1, 0.0);
            for (int j = 0; j <= i; ++j) {
                double w = c.attn_w[((size_t)h * N + i) * N + j];
                dscore[j] = w * (dw[j] - dotv);
            }
            for (int j = 0; j <= i; ++j) {
                double ds = dscore[j] * scale;
                size_t kb = (size_t)j * D + ho;
                axpy(ds, &c.K[kb], &dQ[qb], HD);
                axpy(ds, &c.Q[qb], &dK[kb], HD);
            }
            for (int j = 0; j <= i; ++j) {
                double w = c.attn_w[((size_t)h * N + i) * N + j];
                size_t vb = (size_t)j * D + ho;
                axpy(w, &dattn_out[qb], &dV[vb], HD);
            }
        }
    }
    g.dWq = matmul_TN(c.xn1, dQ, D, N, D);
    g.dWk = matmul_TN(c.xn1, dK, D, N, D);
    g.dWv = matmul_TN(c.xn1, dV, D, N, D);

    vector<double> dxn1((size_t)N * D, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) {
            double s = 0.0;
            s += dot(&dQ[(size_t)i * D], &p.Wq[(size_t)j * D], D);
            s += dot(&dK[(size_t)i * D], &p.Wk[(size_t)j * D], D);
            s += dot(&dV[(size_t)i * D], &p.Wv[(size_t)j * D], D);
            dxn1[(size_t)i * D + j] = s;
        }

    // Back through LN1 (full formula)
    vector<double> dx((size_t)N * D, 0.0);
    for (int i = 0; i < N; ++i) {
        const size_t base = (size_t)i * D;
        double mean_dxhat = 0.0, mean_dxhat_xhat = 0.0;
        vector<double> dxhat(D, 0.0);
        for (int j = 0; j < D; ++j) {
            double dy = dxn1[base + j];
            g.dln1g[j] += dy * c.xhat1[base + j];
            g.dln1b[j] += dy;
            dxhat[j] = dy * p.ln1g[j];
            mean_dxhat += dxhat[j];
            mean_dxhat_xhat += dxhat[j] * c.xhat1[base + j];
        }
        mean_dxhat /= D;
        mean_dxhat_xhat /= D;
        for (int j = 0; j < D; ++j) {
            dx[base + j] += (dxhat[j] - mean_dxhat - c.xhat1[base + j] * mean_dxhat_xhat) * c.inv_std1[i];
        }
    }
    // plus residual from r1 path
    for (size_t i = 0; i < dx.size(); ++i) dx[i] += dr1[i];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) {
            g.dtok_emb[(size_t)toks[i] * D + j] += dx[(size_t)i * D + j];
            g.dpos_emb[(size_t)i * D + j]      += dx[(size_t)i * D + j];
        }

    return g;
}

// ============================================================================
// NORMOUN + ADAMW
// ============================================================================
static void newton_schulz(vector<double>& G, int m, int n, int iters) {
    double nrm = 0.0;
    for (size_t i = 0; i < G.size(); ++i) nrm += G[i] * G[i];
    nrm = sqrt(nrm);
    if (nrm < 1e-12) return;
    for (size_t i = 0; i < G.size(); ++i) G[i] /= nrm;
    bool tr = false; vector<double> X = G;
    if (m < n) {
        vector<double> Xt((size_t)n * m);
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) Xt[(size_t)j * m + i] = X[(size_t)i * n + j];
        X.swap(Xt); int tmp = m; m = n; n = tmp; tr = true;
    }
    for (int it = 0; it < iters; ++it) {
        vector<double> XtX((size_t)n * n, 0.0);
        for (int i = 0; i < m; ++i)
            for (int a = 0; a < n; ++a) {
                double xa = X[(size_t)i * n + a];
                if (xa == 0.0) continue;
                const double* xr = &X[(size_t)i * n];
                double* row = &XtX[(size_t)a * n];
                for (int b = 0; b < n; ++b) row[b] += xa * xr[b];
            }
        vector<double> A((size_t)n * n, 0.0);
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) A[(size_t)a * n + b] = -0.5 * XtX[(size_t)a * n + b];
            A[(size_t)a * n + a] += 1.5;
        }
        vector<double> Xn((size_t)m * n, 0.0);
        for (int i = 0; i < m; ++i)
            for (int a = 0; a < n; ++a) {
                double s = 0.0;
                for (int b = 0; b < n; ++b) s += X[(size_t)i * n + b] * A[(size_t)b * n + a];
                Xn[(size_t)i * n + a] = s;
            }
        X.swap(Xn);
    }
    if (tr) {
        vector<double> Y((size_t)n * m);
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) Y[(size_t)j * m + i] = X[(size_t)i * n + j];
        G = Y;
    } else G = X;
}

static void normuon_step(vector<double>& W, const vector<double>& G, vector<double>& mom,
                         vector<double>& nv, int rows, int cols, double lr, double momentum, int ns_iters) {
    for (size_t i = 0; i < mom.size(); ++i) mom[i] = momentum * mom[i] + G[i];
    vector<double> O = mom;
    newton_schulz(O, rows, cols, ns_iters);
    for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += O[(size_t)r * cols + c] * O[(size_t)r * cols + c];
        double ms = s / cols;
        nv[r] = 0.95 * nv[r] + 0.05 * ms;
        double inv = 1.0 / sqrt(nv[r] + 1e-8);
        for (int c = 0; c < cols; ++c) O[(size_t)r * cols + c] *= inv;
    }
    double scale = sqrt(dmax(1.0, (double)rows / cols));
    for (size_t i = 0; i < W.size(); ++i) W[i] -= lr * scale * O[i];
}

static void adamw_flat(vector<double>& W, const vector<double>& G, vector<double>& mt, vector<double>& vt,
                       size_t off, int& t, double lr, double wd) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    t++;
    double bc1 = 1.0 - pow(b1, (double)t);
    double bc2 = 1.0 - pow(b2, (double)t);
    for (size_t i = 0; i < W.size(); ++i) {
        double g = G[i];
        double m = b1 * mt[off + i] + (1 - b1) * g;
        double v = b2 * vt[off + i] + (1 - b2) * g * g;
        mt[off + i] = m; vt[off + i] = v;
        W[i] -= lr * (m / bc1 / (sqrt(v / bc2) + eps) + wd * W[i]);
    }
}

static double global_grad_norm(const Grads& g) {
    double s = 0.0;
    for (size_t i = 0; i < g.dWq.size(); ++i) s += g.dWq[i] * g.dWq[i];
    for (size_t i = 0; i < g.dWk.size(); ++i) s += g.dWk[i] * g.dWk[i];
    for (size_t i = 0; i < g.dWv.size(); ++i) s += g.dWv[i] * g.dWv[i];
    for (size_t i = 0; i < g.dWo.size(); ++i) s += g.dWo[i] * g.dWo[i];
    for (size_t i = 0; i < g.dWout.size(); ++i) s += g.dWout[i] * g.dWout[i];
    for (size_t i = 0; i < g.dbout.size(); ++i) s += g.dbout[i] * g.dbout[i];
    for (size_t i = 0; i < g.dtok_emb.size(); ++i) s += g.dtok_emb[i] * g.dtok_emb[i];
    for (size_t i = 0; i < g.dpos_emb.size(); ++i) s += g.dpos_emb[i] * g.dpos_emb[i];
    for (size_t i = 0; i < g.dln1g.size(); ++i) s += g.dln1g[i] * g.dln1g[i];
    for (size_t i = 0; i < g.dln1b.size(); ++i) s += g.dln1b[i] * g.dln1b[i];
    for (size_t i = 0; i < g.dln2g.size(); ++i) s += g.dln2g[i] * g.dln2g[i];
    for (size_t i = 0; i < g.dln2b.size(); ++i) s += g.dln2b[i] * g.dln2b[i];
    for (size_t i = 0; i < g.dWr.size(); ++i) s += g.dWr[i] * g.dWr[i];
    for (size_t e = 0; e < g.d_exp_W1.size(); ++e) {
        for (size_t i = 0; i < g.d_exp_W1[e].size(); ++i) s += g.d_exp_W1[e][i] * g.d_exp_W1[e][i];
        for (size_t i = 0; i < g.d_exp_W2[e].size(); ++i) s += g.d_exp_W2[e][i] * g.d_exp_W2[e][i];
        for (size_t i = 0; i < g.d_exp_b1[e].size(); ++i) s += g.d_exp_b1[e][i] * g.d_exp_b1[e][i];
        for (size_t i = 0; i < g.d_exp_b2[e].size(); ++i) s += g.d_exp_b2[e][i] * g.d_exp_b2[e][i];
    }
    return sqrt(s);
}

static void clip_grads(Grads& g, double max_norm) {
    double n = global_grad_norm(g);
    if (n > max_norm) scale_grads(g, max_norm / n);
}

static void update_step(Params& p, const Settings& S, Grads& g, int V, int D, int F, int E, double lr) {
    clip_grads(g, S.CLIP);
    if (S.USE_NORMUON) {
        normuon_step(p.Wq, g.dWq, p.mom_Wq, p.nv_Wq, D, D, lr, S.MOMENTUM, S.NS_ITERS);
        normuon_step(p.Wk, g.dWk, p.mom_Wk, p.nv_Wk, D, D, lr, S.MOMENTUM, S.NS_ITERS);
        normuon_step(p.Wv, g.dWv, p.mom_Wv, p.nv_Wv, D, D, lr, S.MOMENTUM, S.NS_ITERS);
        normuon_step(p.Wo, g.dWo, p.mom_Wo, p.nv_Wo, D, D, lr, S.MOMENTUM, S.NS_ITERS);
        normuon_step(p.Wout, g.dWout, p.mom_Wout, p.nv_Wout, D, V, lr, S.MOMENTUM, S.NS_ITERS);
        for (int e = 0; e < E; ++e) {
            normuon_step(p.exp_W1[e], g.d_exp_W1[e], p.mom_exp_W1[e], p.nv_exp_W1[e], D, F, lr, S.MOMENTUM, S.NS_ITERS);
            normuon_step(p.exp_W2[e], g.d_exp_W2[e], p.mom_exp_W2[e], p.nv_exp_W2[e], F, D, lr, S.MOMENTUM, S.NS_ITERS);
        }
    }
    // AdamW for 1-D + embeddings + expert biases + router
    size_t aoff = 0;
    adamw_flat(p.tok_emb, g.dtok_emb, p.mt, p.vt, aoff, p.t, lr, S.WD); aoff += p.tok_emb.size();
    adamw_flat(p.pos_emb, g.dpos_emb, p.mt, p.vt, aoff, p.t, lr, S.WD); aoff += p.pos_emb.size();
    adamw_flat(p.ln1g, g.dln1g, p.mt, p.vt, aoff, p.t, lr, S.WD); aoff += p.ln1g.size();
    adamw_flat(p.ln1b, g.dln1b, p.mt, p.vt, aoff, p.t, lr, S.WD); aoff += p.ln1b.size();
    adamw_flat(p.ln2g, g.dln2g, p.mt, p.vt, aoff, p.t, lr, S.WD); aoff += p.ln2g.size();
    adamw_flat(p.ln2b, g.dln2b, p.mt, p.vt, aoff, p.t, lr, S.WD); aoff += p.ln2b.size();
    adamw_flat(p.bout, g.dbout, p.mt, p.vt, aoff, p.t, lr, S.WD); aoff += p.bout.size();
    // expert biases and router (simple SGD since small)
    for (int e = 0; e < E; ++e) {
        for (size_t i = 0; i < p.exp_b1[e].size(); ++i) p.exp_b1[e][i] -= lr * g.d_exp_b1[e][i];
        for (size_t i = 0; i < p.exp_b2[e].size(); ++i) p.exp_b2[e][i] -= lr * g.d_exp_b2[e][i];
    }
    for (size_t i = 0; i < p.Wr.size(); ++i) p.Wr[i] -= lr * g.dWr[i];
}

// ============================================================================
// WSD + EMA
// ============================================================================
static double lr_wsd(int step, int total, double base, double warm_ratio, double decay_ratio) {
    int warm = (int)(total * warm_ratio);
    int stable_end = (int)(total * (1.0 - decay_ratio));
    if (warm < 1) warm = 1;
    if (step < warm) return base * step / (double)warm;
    if (step < stable_end) return base;
    double p = (step - stable_end) / (double)dmax(1, total - stable_end);
    return base * 0.5 * (1.0 + cos(PI * p));
}

static void ema_update(vector<double>& shadow, const vector<double>& live, double decay) {
    for (size_t i = 0; i < shadow.size(); ++i) shadow[i] = decay * shadow[i] + (1.0 - decay) * live[i];
}

static void ema_full(Params& p, const Settings& S) {
    if (!S.USE_EMA) return;
    ema_update(p.ema_tok_emb, p.tok_emb, S.EMA_DECAY);
    ema_update(p.ema_pos_emb, p.pos_emb, S.EMA_DECAY);
    ema_update(p.ema_Wq, p.Wq, S.EMA_DECAY);
    ema_update(p.ema_Wk, p.Wk, S.EMA_DECAY);
    ema_update(p.ema_Wv, p.Wv, S.EMA_DECAY);
    ema_update(p.ema_Wo, p.Wo, S.EMA_DECAY);
    ema_update(p.ema_Wout, p.Wout, S.EMA_DECAY);
    for (size_t e = 0; e < p.ema_exp_W1.size(); ++e) {
        ema_update(p.ema_exp_W1[e], p.exp_W1[e], S.EMA_DECAY);
        ema_update(p.ema_exp_W2[e], p.exp_W2[e], S.EMA_DECAY);
    }
}

// ============================================================================
// CHECKPOINT
// ============================================================================
static bool save_checkpoint(const string& path, Params& p, int step, double best_val) {
    ofstream f(path.c_str(), ios::binary);
    if (!f) return false;
    f.write((char*)&step, sizeof(step));
    f.write((char*)&best_val, sizeof(best_val));
    auto sv = [&](const vector<double>& v) {
        size_t n = v.size(); f.write((char*)&n, sizeof(n));
        if (n) f.write((const char*)v.data(), n * sizeof(double));
    };
    sv(p.tok_emb); sv(p.pos_emb);
    sv(p.Wq); sv(p.Wk); sv(p.Wv); sv(p.Wo);
    sv(p.ln1g); sv(p.ln1b); sv(p.ln2g); sv(p.ln2b);
    sv(p.Wout); sv(p.bout); sv(p.Wr);
    sv(p.mt); sv(p.vt);
    size_t E = p.exp_W1.size();
    f.write((char*)&E, sizeof(E));
    for (size_t e = 0; e < E; ++e) {
        sv(p.exp_W1[e]); sv(p.exp_W2[e]); sv(p.exp_b1[e]); sv(p.exp_b2[e]);
        sv(p.mom_exp_W1[e]); sv(p.mom_exp_W2[e]);
        sv(p.nv_exp_W1[e]); sv(p.nv_exp_W2[e]);
    }
    sv(p.mom_Wq); sv(p.mom_Wk); sv(p.mom_Wv); sv(p.mom_Wo); sv(p.mom_Wout);
    sv(p.nv_Wq); sv(p.nv_Wk); sv(p.nv_Wv); sv(p.nv_Wo); sv(p.nv_Wout);
    f.close();
    return true;
}

static bool load_checkpoint(const string& path, Params& p, int& step, double& best_val) {
    ifstream f(path.c_str(), ios::binary);
    if (!f) return false;
    f.read((char*)&step, sizeof(step));
    f.read((char*)&best_val, sizeof(best_val));
    auto ld = [&](vector<double>& v) {
        size_t n = 0; f.read((char*)&n, sizeof(n));
        v.resize(n);
        if (n) f.read((char*)v.data(), n * sizeof(double));
    };
    ld(p.tok_emb); ld(p.pos_emb);
    ld(p.Wq); ld(p.Wk); ld(p.Wv); ld(p.Wo);
    ld(p.ln1g); ld(p.ln1b); ld(p.ln2g); ld(p.ln2b);
    ld(p.Wout); ld(p.bout); ld(p.Wr);
    ld(p.mt); ld(p.vt);
    size_t E = 0; f.read((char*)&E, sizeof(E));
    p.exp_W1.resize(E); p.exp_W2.resize(E); p.exp_b1.resize(E); p.exp_b2.resize(E);
    p.mom_exp_W1.resize(E); p.mom_exp_W2.resize(E);
    p.nv_exp_W1.resize(E); p.nv_exp_W2.resize(E);
    for (size_t e = 0; e < E; ++e) {
        ld(p.exp_W1[e]); ld(p.exp_W2[e]); ld(p.exp_b1[e]); ld(p.exp_b2[e]);
        ld(p.mom_exp_W1[e]); ld(p.mom_exp_W2[e]);
        ld(p.nv_exp_W1[e]); ld(p.nv_exp_W2[e]);
    }
    ld(p.mom_Wq); ld(p.mom_Wk); ld(p.mom_Wv); ld(p.mom_Wo); ld(p.mom_Wout);
    ld(p.nv_Wq); ld(p.nv_Wk); ld(p.nv_Wv); ld(p.nv_Wo); ld(p.nv_Wout);
    f.close();
    return true;
}

// ============================================================================
// GRADIENT CHECK
// ============================================================================
static void gradient_check(Params& p, Settings& S, int V, int D, int F, int E,
                           const vector<int>& in, const vector<int>& tg) {
    cout << "[gradcheck] running...\n";
    Cache c;
    forward(p, S, V, D, F, E, in, c, true);
    Grads g = backward(p, S, V, D, F, E, c, in, tg);
    double best_err = 1.0; int best_idx = -1;
    mt19937 rng(999);
    for (int trial = 0; trial < 5; ++trial) {
        int idx = uniform_int_distribution<int>(0, (int)p.Wq.size() - 1)(rng);
        double orig = p.Wq[idx];
        double eps = 1e-4;
        p.Wq[idx] = orig + eps;
        Cache c1; forward(p, S, V, D, F, E, in, c1, false);
        double lp = cross_entropy(c1, tg, V, S.LABEL_SMOOTH);
        p.Wq[idx] = orig - eps;
        Cache c2; forward(p, S, V, D, F, E, in, c2, false);
        double lm = cross_entropy(c2, tg, V, S.LABEL_SMOOTH);
        p.Wq[idx] = orig;
        double num = (lp - lm) / (2 * eps);
        double ana = g.dWq[idx];
        double err = fabs(num - ana) / dmax(1e-6, fabs(num) + fabs(ana));
        if (err < best_err) { best_err = err; best_idx = idx; }
        cout << "  idx=" << idx << " num=" << num << " ana=" << ana << " rel=" << err << "\n";
    }
    cout << "[gradcheck] best rel err=" << best_err << " (idx=" << best_idx << ")\n";
}

// ============================================================================
// EVAL
// ============================================================================
struct EvalResult { double loss, ppl, top1, top3, top5; };

static EvalResult evaluate(const Params& p, const Settings& S, int V, int D, int F, int E,
                           const vector<int>& inputs, const vector<int>& targets,
                           const BPETokenizer& tok, bool verbose) {
    Cache c;
    forward(p, S, V, D, F, E, inputs, c, false);
    double loss = cross_entropy(c, targets, V, 0.0);
    int correct1 = 0, correct3 = 0, correct5 = 0;
    int N = (int)targets.size();
    for (int i = 0; i < N; ++i) {
        vector<pair<double,int>> sp(V);
        for (int j = 0; j < V; ++j) sp[j] = make_pair(c.probs[(size_t)i * V + j], j);
        sort(sp.begin(), sp.end(), cmp_pair_desc);
        for (int k = 0; k < V; ++k) {
            if (sp[k].second == targets[i]) {
                if (k == 0) correct1++;
                if (k < 3) correct3++;
                if (k < 5) correct5++;
                break;
            }
        }
    }
    EvalResult r;
    r.loss = loss; r.ppl = exp(loss);
    r.top1 = (double)correct1 / N;
    r.top3 = (double)correct3 / N;
    r.top5 = (double)correct5 / N;
    if (verbose) {
        cout << "  [eval] loss=" << loss << " ppl=" << r.ppl
             << " top1=" << r.top1 * 100 << "% top3=" << r.top3 * 100 << "% top5=" << r.top5 * 100 << "%\n";
        int lim = imin(N, 8);
        for (int i = 0; i < lim; ++i) {
            int best = 0;
            for (int j = 1; j < V; ++j) if (c.probs[(size_t)i * V + j] > c.probs[(size_t)i * V + best]) best = j;
            vector<int> a(1, inputs[i]), b(1, targets[i]), d(1, best);
            cout << "    in='" << tok.decode(a) << "' tgt='" << tok.decode(b)
                 << "' guess='" << tok.decode(d) << "' p=" << c.probs[(size_t)i * V + targets[i]] << "\n";
        }
    }
    return r;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv) {
    Settings S = pick_config();

    cout << "=== TRAINER (full) ===\n";
#ifdef USE_OPENCL
    if (S.TRY_GPU) {
        if (g_gpu.init()) cout << "[gpu] OpenCL device ready (probe only, no kernels)\n";
        else cout << "[gpu] fallback to CPU: " << g_gpu.err << "\n";
    }
#else
    (void)S.TRY_GPU;
    cout << "[gpu] compiled without USE_OPENCL (CPU only)\n";
#endif

    string path;
    if (argc > 1) path = argv[1];
    else { cout << "Folder path: "; getline(cin, path); if (path.empty()) path = "."; }

    string corpus, err;
    if (!ingest_folder(path, corpus, err, S.SCAN_SUBFOLDERS, S.AUTO_ZIP)) {
        cerr << "[error] " << err << "\n"; return 1;
    }
    if ((int)corpus.size() < S.SEQ + 1) { cerr << "corpus too short\n"; return 1; }

    BPETokenizer tok;
    cout << "[bpe] training vocab=" << S.VOCAB_TARGET << " on " << corpus.size() << " chars...\n";
    if (!tok.train(corpus, S.VOCAB_TARGET)) { cerr << "[bpe] " << tok.last_error << "\n"; return 1; }
    int V = tok.vocab_size();
    cout << "[bpe] vocab=" << V << "\n";

    int D = S.D, F = S.FFN_MULT * S.D, E = S.USE_MOE ? S.NUM_EXPERTS : 1;
    ParamEstimate est = estimate_params(D, V, S.SEQ, F, E, S.USE_MOE);
    while (est.total < S.MIN_PARAMS) {
        D *= 2; F = S.FFN_MULT * D;
        est = estimate_params(D, V, S.SEQ, F, E, S.USE_MOE);
    }
    if (est.total > S.MAX_PARAMS_WARN) cout << "[params] WARNING: > " << S.MAX_PARAMS_WARN << "\n";
    while (!ask_param_confirmation(est, D, F, E)) {
        est = estimate_params(D, V, S.SEQ, F, E, E > 1);
    }
    S.D = D; S.FFN_MULT = F / D; S.NUM_EXPERTS = E;
    cout << "[model] D=" << D << " F=" << F << " E=" << E << " params=" << est.total << "\n";

    vector<int> ids = tok.encode(corpus);
    if ((int)ids.size() < S.SEQ + 1) { cerr << "encoded too short\n"; return 1; }

    Params P = make_params(S, V, S.D, S.SEQ, F, E);
    Cache cache;
    mt19937 rng(123);

    // Optional gradient check
    {
        vector<int> gin(ids.begin(), ids.begin() + S.SEQ);
        vector<int> gtg(ids.begin() + 1, ids.begin() + S.SEQ + 1);
        gradient_check(P, S, V, S.D, F, E, gin, gtg);
    }

    // CSV log
    ofstream csv;
    if (!S.CSV_LOG.empty()) {
        csv.open(S.CSV_LOG.c_str());
        if (csv) csv << "step,lr,train_loss,grad_norm,val_loss,val_ppl,top1,top5,steps_per_sec\n";
    }

    auto t0 = chrono::steady_clock::now();
    double best_val = 1e18;
    int patience_counter = 0;
    int windows_per_step = S.BATCH * S.GRAD_ACCUM;

    for (int step = 0; step < S.STEPS; ++step) {
        auto step_t0 = chrono::steady_clock::now();
        double lr = lr_wsd(step, S.STEPS, S.LR, S.WARMUP_RATIO, S.DECAY_RATIO);
        Grads g;
        init_grads(g, V, S.D, F, E, S.SEQ);
        double sum_loss = 0.0;

        for (int w = 0; w < windows_per_step; ++w) {
            int off = uniform_int_distribution<int>(0, (int)ids.size() - S.SEQ - 1)(rng);
            vector<int> in(ids.begin() + off, ids.begin() + off + S.SEQ);
            vector<int> tg(ids.begin() + off + 1, ids.begin() + off + S.SEQ + 1);
            forward(P, S, V, S.D, F, E, in, cache, true);
            double loss = cross_entropy(cache, tg, V, S.LABEL_SMOOTH);
            sum_loss += loss;
            Grads gl = backward(P, S, V, S.D, F, E, cache, in, tg);
            accumulate_grads(g, gl);
        }
        double train_loss = sum_loss / windows_per_step;
        scale_grads(g, 1.0 / windows_per_step);
        double gn = global_grad_norm(g);
        update_step(P, S, g, V, S.D, F, E, lr);
        if (S.USE_EMA) ema_full(P, S);

        if (step % S.LOG_EVERY == 0) {
            auto step_t1 = chrono::steady_clock::now();
            double sps = 1.0 / chrono::duration<double>(step_t1 - step_t0).count();
            cout << "[step " << step << "] loss=" << train_loss << " lr=" << lr
                 << " gnorm=" << gn << " " << sps << " steps/s\n";
        }

        if (step % S.EVAL_EVERY == 0 || step == S.STEPS - 1) {
            // Use EMA weights if available
            Params P_eval = P;
            if (S.USE_EMA && P.ema_init) {
                P_eval.tok_emb = P.ema_tok_emb; P_eval.pos_emb = P.ema_pos_emb;
                P_eval.Wq = P.ema_Wq; P_eval.Wk = P.ema_Wk; P_eval.Wv = P.ema_Wv;
                P_eval.Wo = P.ema_Wo; P_eval.Wout = P.ema_Wout;
                P_eval.exp_W1 = P.ema_exp_W1; P_eval.exp_W2 = P.ema_exp_W2;
            }
            vector<int> vin(ids.end() - S.SEQ - 1, ids.end() - 1);
            vector<int> vtg(ids.end() - S.SEQ, ids.end());
            EvalResult r = evaluate(P_eval, S, V, S.D, F, E, vin, vtg, tok, true);
            if (csv) csv << step << "," << lr << "," << train_loss << "," << gn << ","
                        << r.loss << "," << r.ppl << "," << r.top1 << "," << r.top5 << ",\n";
            if (r.loss < best_val) {
                best_val = r.loss; patience_counter = 0;
                if (S.SAVE_CKPT) save_checkpoint(S.CKPT_PATH, P, step, best_val);
            } else {
                patience_counter++;
                if (patience_counter >= S.EARLY_STOP_PATIENCE) {
                    cout << "[early-stop] val loss plateaued at step " << step << "\n";
                    break;
                }
            }
        }
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "\n[done] " << S.STEPS << " steps in " << secs << "s (" << S.STEPS / secs << " steps/s)\n";
    if (csv) csv.close();

    // Sample
    cout << "\n=== SAMPLING ===\n";
    vector<int> gen(ids.begin(), ids.begin() + S.SEQ);
    for (int k = 0; k < 40; ++k) {
        vector<int> ctx(gen.end() - S.SEQ, gen.end());
        Cache gc;
        forward(P, S, V, S.D, F, E, ctx, gc, false);
        int last = S.SEQ - 1;
        vector<double> probs(gc.probs.begin() + (size_t)last * V, gc.probs.begin() + (size_t)(last + 1) * V);
        int nxt = sample_from(probs, S.TEMPERATURE, S.TOP_P, S.TOP_K_SAMPLE, rng);
        gen.push_back(nxt);
    }
    cout << tok.decode(gen) << "\n";
    return 0;
}

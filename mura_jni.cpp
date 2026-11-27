#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include "llama.h"

static llama_model*   g_model = nullptr;
static llama_context* g_ctx   = nullptr;
static llama_sampler* g_smpl  = nullptr;
static int            g_nctx  = 0;
static int            g_past  = 0;

// طول أطول بادئة UTF-8 مكتملة (حتى لا يتعطل JNI مع العربي)
static size_t valid_utf8_len(const std::string& s) {
    size_t n = s.size(), i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        unsigned char c = s[i - 1];
        if ((c & 0xC0) == 0x80) { i--; back++; continue; }
        size_t need = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        size_t have = n - (i - 1);
        return have >= need ? n : i - 1;
    }
    return n;
}

static void free_all() {
    if (g_smpl)  { llama_sampler_free(g_smpl);  g_smpl = nullptr; }
    if (g_ctx)   { llama_free(g_ctx);           g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model);   g_model = nullptr; }
    g_past = 0;
}

static void make_sampler(float temp, float top_p) {
    if (g_smpl) llama_sampler_free(g_smpl);
    g_smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_smpl, llama_sampler_init_penalties(64, 1.1f, 0.f, 0.f));
    if (temp <= 0.f) {
        llama_sampler_chain_add(g_smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(g_smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(g_smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(g_smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(g_smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_gotplx_mura_ai_MuraNative_nLoad(JNIEnv* env, jclass, jstring jpath, jint threads, jint nctx) {
    free_all();
    llama_backend_init();
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.use_mmap = true;
    g_model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) return JNI_FALSE;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nctx; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) { free_all(); return JNI_FALSE; }
    g_nctx = nctx;
    make_sampler(0.7f, 0.9f);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_gotplx_mura_ai_MuraNative_nReset(JNIEnv*, jclass) {
    if (g_ctx) llama_kv_self_clear(g_ctx);
    g_past = 0;
}

JNIEXPORT void JNICALL
Java_com_gotplx_mura_ai_MuraNative_nFree(JNIEnv*, jclass) { free_all(); }

static volatile bool g_stop = false;
JNIEXPORT void JNICALL
Java_com_gotplx_mura_ai_MuraNative_nStop(JNIEnv*, jclass) { g_stop = true; }

// يستقبل نص الدور الجديد فقط (الـ KV cache محفوظ بين الرسائل)
JNIEXPORT jint JNICALL
Java_com_gotplx_mura_ai_MuraNative_nGenerate(JNIEnv* env, jclass, jstring jprompt,
        jint maxNew, jfloat temp, jfloat topP, jobject cb) {
    if (!g_ctx) return -1;
    g_stop = false;
    make_sampler(temp, topP);
    jclass cls = env->GetObjectClass(cb);
    jmethodID onTok = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");

    const char* p = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt(p);
    env->ReleaseStringUTFChars(jprompt, p);

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, false, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), n, false, true);

    if (g_past + n + maxNew > g_nctx) return -2; // السياق ممتلئ: استدعِ nReset

    // معالجة البرومبت على دفعات
    for (int i = 0; i < n; i += 512) {
        int c = std::min(512, n - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, c);
        if (llama_decode(g_ctx, b)) return -3;
        g_past += c;
    }

    std::string pending;
    int produced = 0;
    for (; produced < maxNew && !g_stop; produced++) {
        llama_token t = llama_sampler_sample(g_smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, t)) {
            llama_batch b = llama_batch_get_one(&t, 1); // يُسجَّل <|im_end|> في السياق
            llama_decode(g_ctx, b); g_past++;
            break;
        }
        char buf[256];
        int len = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (len > 0) pending.append(buf, len);
        size_t ok = valid_utf8_len(pending);
        if (ok > 0) {
            std::string out = pending.substr(0, ok);
            pending.erase(0, ok);
            jstring js = env->NewStringUTF(out.c_str());
            env->CallVoidMethod(cb, onTok, js);
            env->DeleteLocalRef(js);
        }
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(g_ctx, b)) return -3;
        g_past++;
    }
    return produced;
}

} // extern "C"

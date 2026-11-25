// libchat: غلاف JNI ثابت فوق llama.cpp — عام لأي نموذج (قالب المحادثة يُقرأ من ملف GGUF نفسه)
#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>
#include "llama.h"

static llama_model*   g_model = nullptr;
static llama_context* g_ctx   = nullptr;
static llama_sampler* g_smpl  = nullptr;
static std::atomic<bool> g_stop{false};
static std::vector<llama_token> g_cache;      // التوكنات الموجودة حالياً في KV cache
static int   g_nctx = 2048, g_topk = 40;
static float g_temp = 0.7f, g_topp = 0.9f, g_pen = 1.1f;
static const int N_BATCH = 512;

static void make_sampler() {
    if (g_smpl) llama_sampler_free(g_smpl);
    g_smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_smpl, llama_sampler_init_penalties(64, g_pen, 0.0f, 0.0f));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_top_k(g_topk));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_top_p(g_topp, 1));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_temp(g_temp));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

static void clear_memory() {
    if (g_ctx) llama_memory_clear(llama_get_memory(g_ctx), true);
    g_cache.clear();
}

static std::string jstr(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r(c);
    env->ReleaseStringUTFChars(s, c);
    return r;
}

static std::string apply_template(const std::vector<std::string>& roles, const std::vector<std::string>& contents) {
    std::vector<llama_chat_message> msgs;
    for (size_t i = 0; i < roles.size(); i++) msgs.push_back({roles[i].c_str(), contents[i].c_str()});
    const char* tmpl = llama_model_chat_template(g_model, nullptr);
    std::vector<char> buf(32768);
    for (int attempt = 0; attempt < 3; attempt++) {
        const char* t = (attempt == 0 && tmpl) ? tmpl : "chatml";
        int n = llama_chat_apply_template(t, msgs.data(), msgs.size(), true, buf.data(), (int)buf.size());
        if (n > (int)buf.size()) { buf.resize(n + 1); n = llama_chat_apply_template(t, msgs.data(), msgs.size(), true, buf.data(), (int)buf.size()); }
        if (n >= 0) return std::string(buf.data(), n);
    }
    return "";
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gotplx_mura_ai_LlamaBridge_nativeLoad(JNIEnv* env, jobject, jstring jpath, jint nctx, jint threads) {
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_cache.clear();
    std::string path = jstr(env, jpath);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path.c_str(), mp);
    if (!g_model) return JNI_FALSE;
    g_nctx = nctx;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nctx;
    cp.n_batch = N_BATCH;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) return JNI_FALSE;
    make_sampler();
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gotplx_mura_ai_LlamaBridge_nativeInfo(JNIEnv* env, jobject) {
    if (!g_model) return env->NewStringUTF("");
    char desc[256];
    llama_model_desc(g_model, desc, sizeof(desc));
    std::string s = std::string(desc) + " · " + std::to_string((double)llama_model_n_params(g_model) / 1e9).substr(0, 4) + "B";
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_gotplx_mura_ai_LlamaBridge_nativeConfigure(JNIEnv*, jobject, jfloat temp, jint topk, jfloat topp, jfloat pen) {
    g_temp = temp; g_topk = topk; g_topp = topp; g_pen = pen;
    if (g_ctx) make_sampler();
}

extern "C" JNIEXPORT void JNICALL
Java_com_gotplx_mura_ai_LlamaBridge_nativeSetThreads(JNIEnv*, jobject, jint n) {
    if (g_ctx) llama_set_n_threads(g_ctx, n, n);
}

extern "C" JNIEXPORT void JNICALL
Java_com_gotplx_mura_ai_LlamaBridge_nativeReset(JNIEnv*, jobject) { clear_memory(); }

extern "C" JNIEXPORT void JNICALL
Java_com_gotplx_mura_ai_LlamaBridge_nativeStop(JNIEnv*, jobject) { g_stop = true; }

extern "C" JNIEXPORT void JNICALL
Java_com_gotplx_mura_ai_LlamaBridge_nativeFree(JNIEnv*, jobject) {
    if (g_smpl)  { llama_sampler_free(g_smpl); g_smpl = nullptr; }
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_cache.clear();
}

// يرجع: 0 تم، -1 المحادثة أطول من السياق (لم يتغير شيء)، -2 خطأ
extern "C" JNIEXPORT jint JNICALL
Java_com_gotplx_mura_ai_LlamaBridge_nativeChat(JNIEnv* env, jobject, jobjectArray jroles, jobjectArray jcontents,
                                               jint maxTokens, jobject cb) {
    g_stop = false;
    if (!g_ctx || !g_model) return -2;

    std::vector<std::string> roles, contents;
    jsize cnt = env->GetArrayLength(jroles);
    for (jsize i = 0; i < cnt; i++) {
        jstring r = (jstring)env->GetObjectArrayElement(jroles, i);
        jstring c = (jstring)env->GetObjectArrayElement(jcontents, i);
        roles.push_back(jstr(env, r));
        contents.push_back(jstr(env, c));
        env->DeleteLocalRef(r);
        env->DeleteLocalRef(c);
    }
    std::string text = apply_template(roles, contents);
    if (text.empty()) return -2;

    jclass cls = env->GetObjectClass(cb);
    jmethodID onBytes = env->GetMethodID(cls, "onBytes", "([B)V");
    jmethodID onProg  = env->GetMethodID(cls, "onProgress", "(II)V");

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    int n = -llama_tokenize(vocab, text.c_str(), (int)text.size(), nullptr, 0, false, true);
    if (n <= 0) return -2;
    std::vector<llama_token> toks(n);
    if (llama_tokenize(vocab, text.c_str(), (int)text.size(), toks.data(), n, false, true) < 0) return -2;
    if (n + maxTokens + 8 > g_nctx) return -1;

    // إعادة استخدام ما هو مشترك مع الـ cache (المحادثة السابقة) وقراءة الجديد فقط
    size_t common = 0;
    while (common < g_cache.size() && common < (size_t)n && g_cache[common] == toks[common]) common++;
    if (common == (size_t)n) common = n - 1;
    if (!llama_memory_seq_rm(llama_get_memory(g_ctx), 0, (llama_pos)common, -1)) { clear_memory(); common = 0; }
    g_cache.resize(common);
    llama_sampler_reset(g_smpl);

    int total = n - (int)common;
    int done = 0;
    while ((int)common + done < n) {
        if (g_stop) return 0;
        int c = std::min(N_BATCH, n - (int)common - done);
        llama_batch b = llama_batch_get_one(toks.data() + common + done, c);
        if (llama_decode(g_ctx, b) != 0) { clear_memory(); return -2; }
        for (int i = 0; i < c; i++) g_cache.push_back(toks[common + done + i]);
        done += c;
        env->CallVoidMethod(cb, onProg, done, total);
    }

    for (int i = 0; i < maxTokens && !g_stop; i++) {
        llama_token id = llama_sampler_sample(g_smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        char buf[256];
        int len = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (len > 0) {
            jbyteArray arr = env->NewByteArray(len);
            env->SetByteArrayRegion(arr, 0, len, (const jbyte*)buf);
            env->CallVoidMethod(cb, onBytes, arr);
            env->DeleteLocalRef(arr);
        }
        llama_batch b = llama_batch_get_one(&id, 1);
        if (llama_decode(g_ctx, b) != 0) { clear_memory(); return -2; }
        g_cache.push_back(id);
    }
    return 0;
}

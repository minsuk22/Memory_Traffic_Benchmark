// JNI entry point used by the Android app (com.memtraffic.bench.MainActivity).
#include <android/log.h>
#include <jni.h>

#include <sstream>

#include "membench.h"

extern "C" JNIEXPORT jint JNICALL
Java_com_memtraffic_bench_MainActivity_nativeRun(JNIEnv* env, jobject thiz, jstring jargs) {
    const char* cargs = env->GetStringUTFChars(jargs, nullptr);
    std::istringstream ss(cargs ? cargs : "");
    env->ReleaseStringUTFChars(jargs, cargs);
    std::vector<std::string> args;
    for (std::string a; ss >> a;) args.push_back(a);

    jclass cls = env->GetObjectClass(thiz);
    jmethodID onLog = env->GetMethodID(cls, "onLog", "(Ljava/lang/String;)V");
    auto log = [&](const std::string& line) {
        __android_log_print(ANDROID_LOG_INFO, "membench", "%s", line.c_str());
        jstring s = env->NewStringUTF(line.c_str());
        env->CallVoidMethod(thiz, onLog, s);
        env->DeleteLocalRef(s);
    };

    mb::Options o;
    std::string err;
    if (!mb::parseArgs(args, o, err) || o.help) {
        log(err);
        log(mb::usage());
        return 2;
    }
    return mb::run(o, log);
}

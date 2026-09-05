// M4 Conversation measurement client. The runtime is unmodified LiteRT-LM v0.13.1.
// Build against c/engine.h and the Android C API shared library.
#include "c/engine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>

static volatile std::sig_atomic_t interrupted = 0;
static void Interrupt(int) { interrupted = 1; }

static int64_t Now() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return int64_t(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

static std::string Quote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c < 32) {
      char escaped[7];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
      out += escaped;
    } else out += c;
  }
  return out + "\"";
}

static std::string Read(const std::string& path) {
  std::ifstream in(path);
  if (!in) return "";
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

class Log {
 public:
  explicit Log(const char* path) : out_(path) {
    if (!out_) { std::cerr << "Cannot open event log\n"; std::exit(2); }
  }
  void Write(const std::string& event, int request = -1,
             const std::string& fields = "", int64_t ns = 0) {
    if (!ns) ns = Now();
    std::lock_guard<std::mutex> guard(lock_);
    out_ << "{\"event\":" << Quote(event) << ",\"ns\":" << ns
         << ",\"request\":" << request << fields << "}\n";
  }
  void Flush() { std::lock_guard<std::mutex> guard(lock_); out_.flush(); }
 private:
  std::mutex lock_;
  std::ofstream out_;
};

struct CallbackState {
  Log* log;
  int request;
  std::mutex lock;
  std::condition_variable cv;
  bool done = false;
  bool had_text = false;
  std::string error;
};

static void Callback(void* opaque, const char* chunk, bool final, const char* error) {
  const int64_t entered = Now();
  auto& s = *static_cast<CallbackState*>(opaque);
  std::lock_guard<std::mutex> guard(s.lock);
  const std::string text = chunk ? chunk : "";
  s.log->Write("callback", s.request,
               ",\"final\":" + std::string(final ? "true" : "false") +
               ",\"text\":" + Quote(text) +
               ",\"error\":" + (error ? Quote(error) : "null"), entered);
  if (!text.empty()) {
    s.had_text = true;
  }
  if (final) {
    s.done = true;
    s.error = error ? error : "";
    s.cv.notify_one();
  }
}


static void Metrics(Log& log, LiteRtLmConversation* conversation, int request) {
  auto* info = litert_lm_conversation_get_benchmark_info(conversation);
  if (!info) { log.Write("metrics_missing", request); return; }
  for (int decode=0; decode<=1; ++decode) {
    const int turns = decode ? litert_lm_benchmark_info_get_num_decode_turns(info)
                             : litert_lm_benchmark_info_get_num_prefill_turns(info);
    for (int i=0; i<turns; ++i) {
      int count=decode ? litert_lm_benchmark_info_get_decode_token_count_at(info,i)
                       : litert_lm_benchmark_info_get_prefill_token_count_at(info,i);
      double rate=decode ? litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info,i)
                         : litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info,i);
      std::ostringstream fields; fields.precision(17);
      fields << ",\"kind\":" << Quote(decode ? "decode" : "prefill")
             << ",\"turn\":" << i << ",\"tokens\":" << count
             << ",\"tokens_per_second\":" << rate;
      log.Write("benchmark_turn",request,fields.str());
    }
  }
  litert_lm_benchmark_info_delete(info);
}

int main(int argc, char** argv) {
  if (argc==2 && std::string(argv[1])=="--clock") { std::cout<<Now()<<"\n"; return 0; }
  if (argc!=7) { std::cerr<<"Usage: m4_observe MODEL LLM_BACKEND VISION_BACKEND CASES_TSV EVENTS SAMPLE_MS\n"; return 2; }
  std::signal(SIGTERM,Interrupt); std::signal(SIGINT,Interrupt);
  const int sample_ms=std::stoi(argv[6]);
  if (sample_ms<0) return 2;
  Log log(argv[5]); std::atomic<bool> stop{false}; std::atomic<int> request{-1};
  log.Write("process_start",-1,",\"pid\":"+std::to_string(getpid())); log.Flush();
  std::thread memory;
  if(sample_ms) memory=std::thread([&]{ while(!stop) {
    auto start=Now(); auto data=Read("/proc/self/status");
    log.Write("memory_sample",request,",\"status_raw\":"+Quote(data)+",\"read_end_ns\":"+std::to_string(Now()),start);
    std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
  }});
  auto snapshot=[&](const std::string& stage) {
    auto start=Now(); auto raw=Read("/proc/self/smaps_rollup");
    log.Write("memory_boundary",request,",\"boundary\":"+Quote(stage)+",\"smaps_raw\":"+Quote(raw)+",\"read_end_ns\":"+std::to_string(Now()),start);
  };
  auto finish=[&](int code) { stop=true; if(memory.joinable()) memory.join();
    log.Write("process_end",request,",\"exit_code\":"+std::to_string(code)); log.Flush(); return code; };
  snapshot("before_load"); log.Write("load_start");
  auto* settings=litert_lm_engine_settings_create(argv[1],argv[2],argv[3],nullptr);
  if(!settings) return finish(3);
  litert_lm_engine_settings_set_max_num_tokens(settings,4096);
  litert_lm_engine_settings_set_max_num_images(settings,1);
  litert_lm_engine_settings_set_cache_dir(settings,"/data/local/tmp/litertlm/m4-cache");
  litert_lm_engine_settings_set_enable_speculative_decoding(settings,false);
  litert_lm_engine_settings_enable_benchmark(settings);
  litert_lm_engine_settings_set_num_prefill_tokens(settings,0);
  litert_lm_engine_settings_set_num_decode_tokens(settings,0);
  auto* engine=litert_lm_engine_create(settings); litert_lm_engine_settings_delete(settings);
  if(!engine) return finish(3);
  log.Write("load_end"); snapshot("after_load");
  std::ifstream cases(argv[4]); std::string line; int result=0, index=0;
  // Each line: case id TAB visual token budget TAB message JSON file.
  while(std::getline(cases,line) && !interrupted) {
    std::istringstream row(line); std::string id,budget,path;
    if(!std::getline(row,id,'\t') || !std::getline(row,budget,'\t') || !std::getline(row,path)) {result=2; break;}
    request=index++; snapshot("before_request");
    log.Write("request_start",request,",\"case\":"+Quote(id)+",\"visual_token_budget\":"+budget);
    auto message=Read(path); if(message.empty()) {result=2;break;}
    auto* sc=litert_lm_session_config_create();
    litert_lm_session_config_set_max_output_tokens(sc,256);
    LiteRtLmSamplerParams sampler{kLiteRtLmSamplerTypeTopP,1,1.0f,1.0f,42};
    litert_lm_session_config_set_sampler_params(sc,&sampler);
    auto* cc=litert_lm_conversation_config_create();
    litert_lm_conversation_config_set_session_config(cc,sc);
    auto* conv=litert_lm_conversation_create(engine,cc);
    litert_lm_conversation_config_delete(cc); litert_lm_session_config_delete(sc);
    if(!conv) {result=3; break;}
    auto* opt=litert_lm_conversation_optional_args_create();
    litert_lm_conversation_optional_args_set_visual_token_budget(opt,std::stoi(budget));
    CallbackState state{&log,request};
    log.Write("send_start",request,",\"message_json\":"+Quote(message));
    int status=litert_lm_conversation_send_message_stream(conv,message.c_str(),nullptr,opt,Callback,&state);
    log.Write("send_return",request,",\"status\":"+std::to_string(status));
    bool timedout=false;
    if(!status) {
      std::unique_lock<std::mutex> lock(state.lock);
      auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(180);
      while(!state.done && !interrupted && std::chrono::steady_clock::now()<deadline)
        state.cv.wait_for(lock,std::chrono::milliseconds(200));
      if(!state.done) {
        timedout=true; log.Write("timeout_or_interrupt",request);
        lock.unlock(); litert_lm_conversation_cancel_process(conv); lock.lock();
        if(!state.cv.wait_for(lock,std::chrono::seconds(60),[&]{return state.done;})) {
          log.Write("cancel_timeout",request); log.Flush(); std::_Exit(4);
        }
      }
    }
    log.Write("request_end",request,",\"status\":"+std::to_string(status)+",\"error\":"+Quote(state.error)+",\"callback_final\":"+(state.done?"true":"false"));
    snapshot("after_request"); Metrics(log,conv,request);
    if(request==0) log.Write("process_maps",request,",\"maps_raw\":"+Quote(Read("/proc/self/maps")));
    litert_lm_conversation_optional_args_delete(opt); litert_lm_conversation_delete(conv);
    snapshot("conversation_released"); log.Flush();
    // Input errors are recorded, then the next case uses a fresh Conversation.
    if(timedout) {result=4;break;}
  }
  if(index==0 || !cases.eof()) result=result ? result : 2;
  litert_lm_engine_delete(engine); log.Write("engine_released"); snapshot("after_release");
  return finish(interrupted ? 5 : result);
}

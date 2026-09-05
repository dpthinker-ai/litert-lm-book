// M3 measurement client. The runtime is unmodified LiteRT-LM v0.13.1.
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
    if (!s.had_text) s.log->Write("first_text", s.request, "", entered);
    s.had_text = true;
  }
  if (final) {
    s.done = true;
    s.error = error ? error : "";
    s.cv.notify_one();
  }
}

static void Metrics(Log& log, LiteRtLmSession* session, int request) {
  auto* info = litert_lm_session_get_benchmark_info(session);
  if (!info) { log.Write("metrics_missing", request); return; }
  for (int decode = 0; decode <= 1; ++decode) {
    const int turns = decode ? litert_lm_benchmark_info_get_num_decode_turns(info)
                             : litert_lm_benchmark_info_get_num_prefill_turns(info);
    for (int i = 0; i < turns; ++i) {
      const int count = decode ? litert_lm_benchmark_info_get_decode_token_count_at(info, i)
                               : litert_lm_benchmark_info_get_prefill_token_count_at(info, i);
      const double rate = decode ? litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, i)
                                 : litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info, i);
      std::ostringstream fields;
      fields.precision(17);
      fields << ",\"kind\":" << Quote(decode ? "decode" : "prefill")
             << ",\"turn\":" << i << ",\"tokens\":" << count
             << ",\"tokens_per_second\":" << rate;
      log.Write("benchmark_turn", request, fields.str());
    }
  }
  litert_lm_benchmark_info_delete(info);
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--clock") {
    std::cout << Now() << "\n"; return 0;
  }
  std::signal(SIGTERM, Interrupt);
  std::signal(SIGINT, Interrupt);
  if (argc != 8) {
    std::cerr << "Usage: m3_observe MODEL BACKEND PROMPT_FILE EVENTS_FILE SECONDS MAX_REQUESTS SAMPLE_MS\n";
    return 2;
  }
  const int seconds = std::stoi(argv[5]);
  const int max_requests = std::stoi(argv[6]);
  const int sample_ms = std::stoi(argv[7]);
  if (seconds < 1 || max_requests < 1 || sample_ms < 0) return 2;
  Log log(argv[4]);
  std::atomic<bool> stop{false};
  std::atomic<int> request{-1};
  std::thread memory;
  log.Write("process_start", -1, ",\"pid\":" + std::to_string(getpid()) +
            ",\"backend\":" + Quote(argv[2]) +
            ",\"sample_ms\":" + std::to_string(sample_ms));
  log.Flush();
  if (sample_ms > 0) {
    memory = std::thread([&] {
      while (!stop) {
        const int64_t started = Now();
        const std::string data = Read("/proc/self/status");
        log.Write("memory_sample", request, ",\"status_raw\":" + Quote(data) +
                  ",\"read_end_ns\":" + std::to_string(Now()), started);
        std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
      }
    });
  }
  auto boundary = [&](const std::string& name) {
    // End events precede memory reads; start events follow them, so API
    // durations do not include the boundary read itself.
    const bool starting = name.size() >= 6 && name.substr(name.size()-6) == "_start";
    if (!starting) log.Write(name, request);
    if (sample_ms > 0) {
      const int64_t started = Now();
      const auto data = Read("/proc/self/smaps_rollup");
      log.Write("memory_boundary", request, ",\"boundary\":" + Quote(name) +
                ",\"smaps_raw\":" + Quote(data) +
                ",\"read_end_ns\":" + std::to_string(Now()), started);
    }
    if (starting) log.Write(name, request);
  };
  auto finish = [&](int code) {
    stop = true;
    if (memory.joinable()) memory.join();
    log.Write("process_end", request, ",\"exit_code\":" + std::to_string(code));
    log.Flush();
    return code;
  };
  boundary("load_start");
  auto* settings = litert_lm_engine_settings_create(argv[1], argv[2], nullptr, nullptr);
  if (!settings) return finish(3);
  litert_lm_engine_settings_set_max_num_tokens(settings, 4096);
  litert_lm_engine_settings_set_cache_dir(settings, "/data/local/tmp/litertlm/m3-cache");
  litert_lm_engine_settings_set_enable_speculative_decoding(settings, false);
  litert_lm_engine_settings_enable_benchmark(settings);
  litert_lm_engine_settings_set_num_prefill_tokens(settings, 0);
  litert_lm_engine_settings_set_num_decode_tokens(settings, 0);
  auto* engine = litert_lm_engine_create(settings);
  litert_lm_engine_settings_delete(settings);
  if (!engine) return finish(3);
  boundary("load_end");
  const int64_t run_start = Now();
  int result = 0;
  for (int i = 0; i < max_requests && !interrupted && Now() - run_start < int64_t(seconds) * 1000000000LL; ++i) {
    request = i;
    log.Write("request_start", i);
    const std::string prompt = Read(argv[3]);
    if (prompt.empty()) { result = 3; break; }
    const std::string message = "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":" + Quote(prompt) + "}]}";
    // Render with the model's own Conversation processor, then execute the
    // resulting text via separate prefill/decode APIs so both boundaries are observable.
    auto* conversation = litert_lm_conversation_create(engine, nullptr);
    if (!conversation) { result = 3; break; }
    const char* rendered_ptr = litert_lm_conversation_render_message_to_string(conversation, message.c_str());
    const std::string rendered = rendered_ptr ? rendered_ptr : "";
    litert_lm_conversation_delete(conversation);
    if (rendered.empty()) { result = 3; break; }
    log.Write("input_prepared", i, ",\"prompt\":" + Quote(prompt) +
              ",\"rendered_prompt\":" + Quote(rendered));
    auto* config = litert_lm_session_config_create();
    if (!config) { result = 3; break; }
    litert_lm_session_config_set_apply_prompt_template(config, false);
    litert_lm_session_config_set_max_output_tokens(config, 512);
    LiteRtLmSamplerParams sampler{kLiteRtLmSamplerTypeTopP, 1, 1.0f, 1.0f, 42};
    litert_lm_session_config_set_sampler_params(config, &sampler);
    auto* session = litert_lm_engine_create_session(engine, config);
    litert_lm_session_config_delete(config);
    if (!session) { result = 3; break; }
    LiteRtLmInputData input{kLiteRtLmInputDataTypeText, rendered.data(), rendered.size()};
    boundary("prefill_api_start");
    int status = litert_lm_session_run_prefill(session, &input, 1);
    boundary("prefill_api_end");
    if (status) { litert_lm_session_delete(session); result = 3; break; }
    CallbackState state{&log, i};
    boundary("decode_api_start");
    status = litert_lm_session_run_decode_async(session, Callback, &state);
    if (status) { litert_lm_session_delete(session); result = 3; break; }
    bool timed_out = false;
    {
      std::unique_lock<std::mutex> lock(state.lock);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
      while (!state.done && !interrupted && std::chrono::steady_clock::now() < deadline)
        state.cv.wait_for(lock, std::chrono::milliseconds(200));
      if (!state.done) {
        timed_out = !interrupted;
        log.Write(interrupted ? "interrupted" : "timeout", i);
        lock.unlock();
        litert_lm_session_cancel_process(session);
        lock.lock();
        if (!state.cv.wait_for(lock, std::chrono::seconds(60), [&] { return state.done; })) {
          log.Write("cancel_timeout", i);
          log.Flush();
          std::_Exit(4);
        }
      }
    }
    boundary("decode_api_end");
    log.Write("request_end", i, ",\"had_text\":" + std::string(state.had_text ? "true" : "false") +
              ",\"error\":" + Quote(state.error));
    Metrics(log, session, i);
    if (i == 0) log.Write("process_maps", i, ",\"maps_raw\":" + Quote(Read("/proc/self/maps")));
    litert_lm_session_delete(session);
    boundary("session_released");
    log.Flush();
    if (timed_out || !state.had_text || !state.error.empty()) { result = 3; break; }
  }
  if (interrupted) result = 5;
  boundary("engine_release_start");
  litert_lm_engine_delete(engine);
  boundary("engine_release_end");
  std::this_thread::sleep_for(std::chrono::seconds(3));
  boundary("post_release_3s");
  return finish(result);
}

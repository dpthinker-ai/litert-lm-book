// Count the non-visual positions for the single-image Gemma 4 test message.
// This diagnostic does not generate output or measure performance.
#include "c/engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  auto* settings=litert_lm_engine_settings_create(argv[1],"gpu","gpu",nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings,4096);
  litert_lm_engine_settings_set_cache_dir(settings,"/data/local/tmp/litertlm/m4-cache");
  auto* engine=litert_lm_engine_create(settings);litert_lm_engine_settings_delete(settings);
  if(!engine)return 3;
  auto* conv=litert_lm_conversation_create(engine,nullptr);if(!conv)return 3;
  std::ifstream f(argv[2]);std::ostringstream b;b<<f.rdbuf();
  const char* rendered=litert_lm_conversation_render_message_to_string(conv,b.str().c_str());
  if(!rendered)return 3;
  std::string prompt=rendered;
  std::ofstream("rendered.txt")<<prompt;
  std::string marker="<|image|>";
  auto pos=prompt.find(marker);
  if(pos==std::string::npos || prompt.find(marker,pos+marker.size())!=std::string::npos)return 3;
  // Gemma4DataProcessor emits three text parts around one image and ImageEnd.
  std::vector<std::string> parts{prompt.substr(0,pos)+"<|image>","\n\n",prompt.substr(pos+marker.size())};
  auto* start=litert_lm_engine_get_start_token(engine);const int* bos_ids=nullptr;size_t bos_count=0;
  std::string bos;
  if(start && litert_lm_token_union_get_type(start)==kLiteRtLmTokenUnionTypeIds) {
    litert_lm_token_union_get_ids(start,&bos_ids,&bos_count);
    auto* decoded=litert_lm_engine_detokenize(engine,bos_ids,bos_count);
    if(decoded){bos=litert_lm_detokenize_result_get_string(decoded);litert_lm_detokenize_result_delete(decoded);}
  }
  // The model normally provides BOS as an id; abort rather than assume a value.
  if(bos.empty() || bos_count!=1)return 3;
  std::cout<<"{\"counts\":[";size_t total=2; // one ImageEnd plus the fresh Session BOS position
  for(size_t i=0;i<parts.size();++i) {
    std::ofstream("part-"+std::to_string(i)+".txt")<<parts[i];
    bool has_bos=parts[i].rfind(bos,0)==0;
    auto text=has_bos?parts[i].substr(bos.size()):parts[i];
    auto* tokens=litert_lm_engine_tokenize(engine,text.c_str());if(!tokens)return 3;
    auto count=litert_lm_tokenize_result_get_num_tokens(tokens)+(has_bos?1:0);total+=count;
    if(i)std::cout<<",";std::cout<<count;
    litert_lm_tokenize_result_delete(tokens);
  }
  std::cout<<"],\"image_end_positions\":1,\"bos_positions\":1,\"nonvisual_positions\":"<<total<<"}\n";
  litert_lm_token_union_delete(start);litert_lm_conversation_delete(conv);litert_lm_engine_delete(engine);
}

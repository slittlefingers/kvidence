// Standalone diagnostic: existing LiteRT C ABI, no APK/runtime modifications.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#include <jni.h>
#include "nlohmann/json.hpp"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

using J=nlohmann::json;
using Bytes=std::vector<uint8_t>;
using Data=std::map<std::string,Bytes>;
#define CHECK(call) do {auto rc=(call);if(rc!=kLiteRtStatusOk)throw std::runtime_error(std::string(#call)+" status="+std::to_string(rc));}while(0)
static void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static Bytes read(const std::string& p){std::ifstream f(p,std::ios::binary);require(bool(f),"Cannot read "+p);return Bytes(std::istreambuf_iterator<char>(f),{});}
static void save(const std::string& p,const Bytes& b){std::ofstream f(p,std::ios::binary);f.write((const char*)b.data(),b.size());require(bool(f),"Cannot write "+p);}
static void savej(const std::string& p,const J& j){std::ofstream f(p);f<<j.dump(2)<<'\n';require(bool(f),"Cannot write "+p);}
template<class T>static Bytes bytes(const std::vector<T>& v){Bytes b(v.size()*sizeof(T));std::memcpy(b.data(),v.data(),b.size());return b;}
static int width(LiteRtElementType t){switch(t){case kLiteRtElementTypeFloat32:case kLiteRtElementTypeInt32:return 4;case kLiteRtElementTypeInt8:return 1;default:throw std::runtime_error("Unsupported element type "+std::to_string(t));}}

struct Buffer {
 LiteRtTensorBuffer h=nullptr;LiteRtRankedTensorType type{};size_t allocation=0,count=1;int elem=0;
 std::vector<size_t> strides;
 Buffer(LiteRtEnvironment env,LiteRtTensor t,LiteRtTensorBufferRequirements req){
  CHECK(LiteRtGetRankedTensorType(t,&type));
  CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env,&type,req,&h));
  CHECK(LiteRtGetTensorBufferTensorType(h,&type));CHECK(LiteRtGetTensorBufferSize(h,&allocation));
  elem=width(type.element_type);strides.resize(type.layout.rank);size_t s=1;
  for(int i=int(type.layout.rank)-1;i>=0;--i){require(type.layout.dimensions[i]>0,"Dynamic/empty shape");strides[i]=type.layout.has_strides?type.layout.strides[i]:s;s*=type.layout.dimensions[i];}
  count=s;size_t last=0;for(unsigned i=0;i<type.layout.rank;i++)last+=(type.layout.dimensions[i]-1)*strides[i];
  require((last+1)*elem<=allocation,"Stride exceeds allocation");
 }
 ~Buffer(){if(h)LiteRtDestroyTensorBuffer(h);}
 size_t offset(size_t linear)const{size_t off=0;for(int d=int(type.layout.rank)-1;d>=0;--d){off+=(linear%type.layout.dimensions[d])*strides[d];linear/=type.layout.dimensions[d];}return off*elem;}
 Bytes get()const{void* p;CHECK(LiteRtLockTensorBuffer(h,&p,kLiteRtTensorBufferLockModeRead));Bytes out(count*elem);for(size_t i=0;i<count;i++)std::memcpy(out.data()+i*elem,(uint8_t*)p+offset(i),elem);CHECK(LiteRtUnlockTensorBuffer(h));return out;}
 void put(const Bytes& b){require(b.size()==count*elem,"Input byte count mismatch");void* p;CHECK(LiteRtLockTensorBuffer(h,&p,kLiteRtTensorBufferLockModeWrite));for(size_t i=0;i<count;i++)std::memcpy((uint8_t*)p+offset(i),b.data()+i*elem,elem);CHECK(LiteRtUnlockTensorBuffer(h));}
 J layout()const{return J{{"shape",std::vector<int32_t>(type.layout.dimensions,type.layout.dimensions+type.layout.rank)},{"strides",strides},{"element_type",int(type.element_type)},{"element_bytes",elem},{"allocated_bytes",allocation}};}
};
struct Section {
 LiteRtEnvironment env;LiteRtModel model=nullptr;LiteRtCompiledModel compiled=nullptr;J meta;
 Section(LiteRtEnvironment e,int fd,const J& m,bool npu):env(e),meta(m){
  CHECK(LiteRtCreateModelFromFd(env,fd,m.at("offset").get<size_t>(),m.at("length").get<size_t>(),&model));
  LiteRtOptions opt;CHECK(LiteRtCreateOptions(&opt));CHECK(LiteRtSetOptionsHardwareAccelerators(opt,npu?kLiteRtHwAcceleratorNpu:kLiteRtHwAcceleratorCpu));
  auto compile_status=LiteRtCreateCompiledModel(env,model,opt,&compiled);LiteRtDestroyOptions(opt);CHECK(compile_status);
 }
 ~Section(){if(compiled)LiteRtDestroyCompiledModel(compiled);if(model)LiteRtDestroyModel(model);}
};
struct Signature {
 Section& sec;LiteRtParamIndex index;std::string name;std::vector<LiteRtTensorBuffer> in,out;
 std::map<std::string,std::unique_ptr<Buffer>> inputs,outputs;
 Signature(Section& s,const std::string& n):sec(s),index(s.meta.at("signatures").at(n).at("index")),name(n){
  LiteRtSignature sig;CHECK(LiteRtGetModelSignature(sec.model,index,&sig));const char* actual;CHECK(LiteRtGetSignatureKey(sig,&actual));require(n==actual,"Signature index mismatch");
  for(bool input:{true,false}){LiteRtParamIndex total;CHECK(input?LiteRtGetNumSignatureInputs(sig,&total):LiteRtGetNumSignatureOutputs(sig,&total));
   for(LiteRtParamIndex i=0;i<total;i++){const char* key;LiteRtTensor t;LiteRtTensorBufferRequirements req;
    CHECK(input?LiteRtGetSignatureInputName(sig,i,&key):LiteRtGetSignatureOutputName(sig,i,&key));
    CHECK(input?LiteRtGetSignatureInputTensorByIndex(sig,i,&t):LiteRtGetSignatureOutputTensorByIndex(sig,i,&t));
    CHECK(input?LiteRtGetCompiledModelInputBufferRequirements(sec.compiled,index,i,&req):LiteRtGetCompiledModelOutputBufferRequirements(sec.compiled,index,i,&req));
    auto b=std::make_unique<Buffer>(sec.env,t,req);(input?in:out).push_back(b->h);(input?inputs:outputs)[key]=std::move(b);
   }
  }
 }
 double run(){double t=now();CHECK(LiteRtRunCompiledModel(sec.compiled,index,in.size(),in.data(),out.size(),out.data()));return now()-t;}
 Data get(bool kv_only=false){Data d;for(auto& [n,b]:outputs)if(!kv_only||n.rfind("kv_slice_",0)==0)d[n]=b->get();return d;}
 void put(const Data& d){for(auto& [n,b]:d)inputs.at(n)->put(b);}
};
static Bytes repeat(const Bytes& src,int times){Bytes b;for(int i=0;i<times;i++)b.insert(b.end(),src.begin(),src.end());return b;}
static Data loadkv(Signature& s,const J& snapshot,const Bytes& payload){
 Data expected;
 for(auto& b:snapshot.at("buffers")){
  std::string n=b.at("name");if(!s.inputs.count(n))continue;auto& dst=*s.inputs.at(n);require(dst.elem==b.at("element_bytes").get<int>(),"KV dtype mismatch");
  auto shape=b.at("shape").get<std::vector<int>>();auto stride=b.at("strides").get<std::vector<size_t>>();
  require(shape==dst.layout().at("shape").get<std::vector<int>>(),"KV shape mismatch");Bytes packed(dst.count*dst.elem);
  for(size_t i=0;i<dst.count;i++){size_t z=i,off=0;for(int d=int(shape.size())-1;d>=0;d--){off+=(z%shape[d])*stride[d];z/=shape[d];}off=off*dst.elem+b.at("payload_offset").get<size_t>();require(off+dst.elem<=payload.size(),"KV payload bounds");std::memcpy(packed.data()+i*dst.elem,payload.data()+off,dst.elem);}
  dst.put(packed);expected[n]=std::move(packed);
 }
 return expected;
}
static Data row(Signature& sig,const Data& all,int q,int batch){
 Data out;
 for(auto& [name,data]:all){auto& b=*sig.outputs.at(name);require(b.elem==1,"Expected INT8 KV output");
  bool key=name.rfind("kv_slice_k_",0)==0;int axis=key?2:3;require(b.type.layout.dimensions[axis]==batch,"Unexpected KV sequence axis");
  size_t inner=1,outer=1;for(unsigned i=axis+1;i<b.type.layout.rank;i++)inner*=b.type.layout.dimensions[i];for(int i=0;i<axis;i++)outer*=b.type.layout.dimensions[i];
  Bytes v(outer*inner);for(size_t o=0;o<outer;o++)std::memcpy(v.data()+o*inner,data.data()+(o*batch+q)*inner,inner);out[name]=std::move(v);
 }return out;
}
static J compare(const Data& a,const Data& b){
 J result=J::object();for(auto& [n,x]:a){auto& y=b.at(n);require(x.size()==y.size(),"Compare size mismatch");double l1=0,l2=0;int mx=0;size_t changed=0;
  for(size_t i=0;i<x.size();i++){int d=int(int8_t(x[i]))-int(int8_t(y[i]));mx=std::max(mx,std::abs(d));l1+=std::abs(d);l2+=d*d;changed+=d!=0;}
  result[n]={{"max_abs_lsb",mx},{"l2_integer",std::sqrt(l2)},{"mean_abs_lsb",l1/x.size()},{"different",changed},{"elements",x.size()}};
 }return result;
}
static void dump(const std::string& dir,const std::string& tag,const Data& d){for(auto& [n,b]:d)save(dir+"/"+tag+"-"+n+".bin",b);}

int main(int argc,char** argv){
 try{
  require(argc==5,"Usage: probe MODEL LIBDIR INPUTDIR OUTPUTDIR");std::string indir=argv[3],outdir=argv[4];
  J job=J::parse(read(indir+"/probe.json"));auto payload=read(indir+"/kv_payload.bin");
  LiteRtEnvOption option{};option.tag=kLiteRtEnvOptionTagDispatchLibraryDir;option.value.type=kLiteRtAnyTypeString;option.value.str_value=argv[2];
  LiteRtEnvironment env;CHECK(LiteRtCreateEnvironment(1,&option,&env));int fd=open(argv[1],O_RDONLY);require(fd>=0,"Open model failed");
  auto& sections=job.at("model_sections");
  std::cout<<"Loading existing NPU decoder"<<std::endl;
  Section decoder(env,fd,sections.at("tf_lite_prefill_decode"),true);
  Section embedder(env,fd,sections.at("tf_lite_embedder"),false);
  Section perlayer(env,fd,sections.at("tf_lite_per_layer_embedder"),false);
  Section aux(env,fd,sections.at("tf_lite_aux"),false);close(fd);
  int batch=job.at("candidates").size();require(batch==4||batch==128,"This probe supports 4 or 128 candidates");std::string signature=batch==4?"verify":"prefill_128";
  Signature dec(decoder,"decode"),ver(decoder,signature),emb(embedder,"decode_embedder"),ple(perlayer,"decode_per_layer_embedder"),rope(aux,"decode_rope"),mask(aux,"decode_mask");
  auto expected_kv=loadkv(dec,job.at("snapshot"),payload);loadkv(ver,job.at("snapshot"),payload);
  J report={{"schema",2},{"backend","NPU"},{"batch",batch},{"signature",signature},{"candidates",job.at("candidates")},{"prefix_source",job.at("source_capture")},{"cases",J::array()}};
  for(auto* s:{&dec,&ver}){for(auto& [n,b]:s->outputs)report["output_layouts"][s->name][n]=b->layout();for(auto& [n,b]:s->inputs)report["input_layouts"][s->name][n]=b->layout();}
  std::vector<Data> embeddings;
  for(auto& c:job.at("candidates")){Bytes id=bytes<int32_t>({c.at("id").get<int>()});emb.inputs.at("token_ids")->put(id);ple.inputs.at("token_ids")->put(id);emb.run();ple.run();embeddings.push_back({{"embeddings",emb.get().at("embeddings")},{"per_layer_embeddings",ple.get().at("embeddings")}});}
  for(int t:job.at("prefix_positions")){
   require(t>=0&&t<=job.at("snapshot").at("kv_positions").get<int>(),"Prefix outside captured KV");
   rope.inputs.at("input_pos")->put(bytes<int32_t>({t}));rope.run();Data positions=rope.get();
   mask.inputs.at("input_tokens")->put(bytes<int32_t>({1437}));mask.inputs.at("time_step")->put(bytes<int32_t>({t}));mask.run();Data masks=mask.get();
   J c={{"prefix_length",t},{"serial_seconds",J::array()},{"batch_seconds",J::array()},{"comparisons",J::array()}};
   dec.put(positions);dec.put(masks);std::vector<Data> serial;
   for(int q=0;q<batch;q++){dec.put(embeddings[q]);dec.run();J times=J::array();for(int n=0;n<5;n++)times.push_back(dec.run());c["serial_seconds"].push_back(times);serial.push_back(dec.get(true));dump(outdir,"p"+std::to_string(t)+"-serial"+std::to_string(q),serial.back());}
   Data vin;
   for(auto& [n,b]:positions)vin[n]=repeat(b,batch);
   for(auto& [n,b]:embeddings[0]){Bytes joined;for(auto& e:embeddings)joined.insert(joined.end(),e.at(n).begin(),e.at(n).end());vin[n]=std::move(joined);}
   for(auto& [n,b]:masks){require(b.size()==4097*sizeof(float),"Unexpected baseline mask");std::vector<float> base(4097);std::memcpy(base.data(),b.data(),b.size());float hidden=base[4095],visible=base[4096];require(hidden!=visible,"Mask values not distinct");std::vector<float> branch(batch*(4096+batch),hidden);
    for(int q=0;q<batch;q++){std::copy(base.begin(),base.begin()+4096,branch.begin()+q*(4096+batch));branch[q*(4096+batch)+4096+q]=visible;}vin[n]=bytes(branch);
   }
   ver.put(vin);ver.run();for(int n=0;n<5;n++)c["batch_seconds"].push_back(ver.run());Data batched=ver.get(true);dump(outdir,"p"+std::to_string(t)+"-batch",batched);
   for(int q=0;q<batch;q++)c["comparisons"].push_back(compare(serial[q],row(ver,batched,q,batch)));
   // Check every row, including later rows which a normal causal mask would
   // allow to see earlier candidates. Fix one row and replace its neighbours.
   c["neighbour_changes"]=J::array();
   std::vector<int> fixed_rows=batch==4?std::vector<int>{0,1,2,3}:std::vector<int>{0,1,63,127};c["isolation_rows"]=fixed_rows;
   for(int fixed:fixed_rows){
    for(auto& [n,b]:embeddings[0]){Bytes changed;for(int q=0;q<batch;q++){auto& e=embeddings[q==fixed?q:(q+1)%batch].at(n);changed.insert(changed.end(),e.begin(),e.end());}ver.inputs.at(n)->put(changed);}
    ver.run();auto isolation=ver.get(true);c["neighbour_changes"].push_back(compare(row(ver,batched,fixed,batch),row(ver,isolation,fixed,batch)));
   }
   // Negative control: deliberately allow earlier candidate slots. This should
   // change deeper-layer results for later candidates, demonstrating mask effect.
   ver.put(vin);
   for(auto& [n,b]:masks){float visible;std::memcpy(&visible,b.data()+4096*sizeof(float),sizeof(float));auto causal=vin.at(n);for(int q=0;q<batch;q++)for(int k=0;k<=q;k++)std::memcpy(causal.data()+(q*(4096+batch)+4096+k)*sizeof(float),&visible,sizeof(float));ver.inputs.at(n)->put(causal);}
   ver.run();auto causal=ver.get(true);c["causal_negative_control"]=compare(row(ver,batched,batch-1,batch),row(ver,causal,batch-1,batch));
   for(auto* sig:{&dec,&ver})for(auto& [n,b]:expected_kv)if(sig->inputs.count(n))require(sig->inputs.at(n)->get()==b,"Input prefix KV changed: "+sig->name+"/"+n);
   c["input_kv_unchanged"]=true;
   report["cases"].push_back(c);savej(outdir+"/report.json",report);std::cout<<"Completed prefix "<<t<<std::endl;
  }
  report["completed"]=true;savej(outdir+"/report.json",report);std::cout<<"PROBE COMPLETE"<<std::endl;
  // The process owns all objects; RAII destroys models before process exit.
  return 0;
 }catch(const std::exception& e){std::cerr<<"PROBE FAILED: "<<e.what()<<std::endl;return 1;}
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_annoymous_CandidateBatchProbeDeviceTest_runProbe(
 JNIEnv* env,jclass,jobjectArray arguments,jstring log_path){
 std::vector<std::string> storage={"candidate_probe"};
 for(jsize i=0;i<env->GetArrayLength(arguments);i++){
  auto value=(jstring)env->GetObjectArrayElement(arguments,i);const char* s=env->GetStringUTFChars(value,nullptr);
  storage.emplace_back(s);env->ReleaseStringUTFChars(value,s);env->DeleteLocalRef(value);
 }
 std::vector<char*> argv;for(auto& s:storage)argv.push_back(s.data());
 const char* path=env->GetStringUTFChars(log_path,nullptr);int log=open(path,O_WRONLY|O_CREAT|O_TRUNC,0600);env->ReleaseStringUTFChars(log_path,path);
 if(log<0)return 90;int old_out=dup(STDOUT_FILENO),old_err=dup(STDERR_FILENO);fflush(nullptr);dup2(log,STDOUT_FILENO);dup2(log,STDERR_FILENO);close(log);
 int rc=main(argv.size(),argv.data());std::cout.flush();std::cerr.flush();fflush(nullptr);dup2(old_out,STDOUT_FILENO);dup2(old_err,STDERR_FILENO);close(old_out);close(old_err);return rc;
}

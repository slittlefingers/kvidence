#pragma once
#include "tensor_copy.h"
// Adapted from the independently validated npu_candidate_batch probe.
// Kept local so the original experiment is unchanged.
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
inline thread_local bool native_optimized=true;
inline thread_local bool native_optimized_v2=true;
inline thread_local std::map<std::string,double> profile_seconds;
inline thread_local std::map<std::string,size_t> profile_counts;
static void measured(const std::string& key,double start){profile_seconds[key]+=now()-start;}
static Bytes read(const std::string& p){std::ifstream f(p,std::ios::binary);require(bool(f),"Cannot read "+p);return Bytes(std::istreambuf_iterator<char>(f),{});}
static void save(const std::string& p,const Bytes& b){std::ofstream f(p,std::ios::binary);f.write((const char*)b.data(),b.size());require(bool(f),"Cannot write "+p);}
static void savej(const std::string& p,const J& j){std::ofstream f(p);f<<j.dump(2)<<'\n';require(bool(f),"Cannot write "+p);}
template<class T>static Bytes bytes(const std::vector<T>& v){Bytes b(v.size()*sizeof(T));std::memcpy(b.data(),v.data(),b.size());return b;}
static int width(LiteRtElementType t){switch(t){case kLiteRtElementTypeFloat32:case kLiteRtElementTypeInt32:return 4;case kLiteRtElementTypeBool:case kLiteRtElementTypeInt8:return 1;default:throw std::runtime_error("Unsupported element type "+std::to_string(t));}}

struct Buffer {
 LiteRtTensorBuffer h=nullptr;LiteRtRankedTensorType type{};size_t allocation=0,count=1;int elem=0;
 std::vector<size_t> strides;
 std::unique_ptr<TensorCopy> copier;
 Buffer(LiteRtEnvironment env,LiteRtTensor t,LiteRtTensorBufferRequirements req){
  CHECK(LiteRtGetRankedTensorType(t,&type));
  CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env,&type,req,&h));
  CHECK(LiteRtGetTensorBufferTensorType(h,&type));CHECK(LiteRtGetTensorBufferSize(h,&allocation));
  elem=width(type.element_type);strides.resize(type.layout.rank);size_t s=1;
  for(int i=int(type.layout.rank)-1;i>=0;--i){require(type.layout.dimensions[i]>0,"Dynamic/empty shape");strides[i]=type.layout.has_strides?type.layout.strides[i]:s;s*=type.layout.dimensions[i];}
  count=s;size_t last=0;for(unsigned i=0;i<type.layout.rank;i++)last+=(type.layout.dimensions[i]-1)*strides[i];
  require((last+1)*elem<=allocation,"Stride exceeds allocation");
  copier=std::make_unique<TensorCopy>(std::vector<size_t>(type.layout.dimensions,type.layout.dimensions+type.layout.rank),strides,elem);
 }
 ~Buffer(){if(h)LiteRtDestroyTensorBuffer(h);}
 size_t offset(size_t linear)const{size_t off=0;for(int d=int(type.layout.rank)-1;d>=0;--d){off+=(linear%type.layout.dimensions[d])*strides[d];linear/=type.layout.dimensions[d];}return off*elem;}
 Bytes get()const{
  double tick=now();Bytes out(count*elem);measured("buffer_read_allocate",tick);void* p;
  tick=now();CHECK(LiteRtLockTensorBuffer(h,&p,kLiteRtTensorBufferLockModeRead));measured("buffer_read_lock",tick);
  tick=now();copier->copy(p,out.data(),0,count,false,native_optimized);measured("buffer_read_copy",tick);
  tick=now();CHECK(LiteRtUnlockTensorBuffer(h));measured("buffer_read_unlock",tick);
  profile_counts["buffer_read_bytes"]+=count*elem;return out;
 }
 void put(const Bytes& b){
  require(b.size()==count*elem,"Input byte count mismatch");void* p;double tick=now();
  CHECK(LiteRtLockTensorBuffer(h,&p,kLiteRtTensorBufferLockModeWrite));measured("buffer_write_lock",tick);
  tick=now();copier->copy(p,const_cast<uint8_t*>(b.data()),0,count,true,native_optimized);measured("buffer_write_copy",tick);
  tick=now();CHECK(LiteRtUnlockTensorBuffer(h));measured("buffer_write_unlock",tick);
  profile_counts["buffer_write_bytes"]+=count*elem;
 }
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
 double run(){double t=now();CHECK(LiteRtRunCompiledModel(sec.compiled,index,in.size(),in.data(),out.size(),out.data()));double elapsed=now()-t;profile_seconds["model_"+name]+=elapsed;profile_counts["model_"+name]++;return elapsed;}
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
 for(auto& [name,data]:all){auto& b=*sig.outputs.at(name);require(b.elem==1||b.elem==4,"Expected INT8/float32 KV output");
  bool key=name.rfind("kv_slice_k_",0)==0;int axis=key?2:3;require(b.type.layout.dimensions[axis]==batch,"Unexpected KV sequence axis");
  size_t inner=1,outer=1;for(unsigned i=axis+1;i<b.type.layout.rank;i++)inner*=b.type.layout.dimensions[i];for(int i=0;i<axis;i++)outer*=b.type.layout.dimensions[i];
  Bytes v(outer*inner*b.elem);for(size_t o=0;o<outer;o++)std::memcpy(v.data()+o*inner*b.elem,data.data()+(o*batch+q)*inner*b.elem,inner*b.elem);out[name]=std::move(v);
 }return out;
}

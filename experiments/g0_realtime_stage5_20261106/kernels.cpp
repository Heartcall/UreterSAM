// Self-contained x64 SSE2 kernel: no CRT, SDK headers, heap or DLL entry point.
// The installed MSVC toolchain lacks Windows SDK/UCRT headers.
typedef unsigned char uint8_t;
typedef union __declspec(intrin_type) __declspec(align(16)) __m128 { float f[4]; } __m128;
extern "C" __m128 _mm_sqrt_ss(__m128);
#pragma intrinsic(_mm_sqrt_ss)
extern "C" int _fltused=0;
static float square_root(float x) {
 __m128 v;v.f[0]=x;v.f[1]=v.f[2]=v.f[3]=0;
 v=_mm_sqrt_ss(v);return v.f[0];
}
static float maximum(float a,float b) {return a>b?a:b;}
static float minimum(float a,float b) {return a<b?a:b;}
static double absolute(double x) {return x<0?-x:x;}

// Compile with /fp:strict: no reassociation or fused multiply-add.
static float sum3(float a,float b,float c) { return (a+b)+c; }
static float norm3(float a,float b,float c) { return square_root(sum3(a*a,b*b,c*c)); }
static float clip(float x) { return minimum(maximum(x,0.0f),1.0f); }

extern "C" __declspec(dllexport) void s2_evidence(
 const float* color,const float* noise,const float* gradient,int size,
 const float* seed,const float* cfg,float* score,uint8_t* strong,uint8_t* weak,uint8_t* uncertain) {
 const float bgblue=seed[5]-0.5f*(seed[3]+seed[4]);
 for(int i=0;i<size;++i) {
  score[i]=0;strong[i]=weak[i]=uncertain[i]=0;
  const float* c=color+3*i;
  float blue=c[2]-0.5f*(c[0]+c[1]);
  float gap=blue-bgblue;
  if(!(blue>cfg[2] && gap>cfg[3] && noise[i]<cfg[6])) continue;
  float v0=c[0]-float(1.0/3.0),v1=c[1]-float(1.0/3.0),v2=c[2]-float(1.0/3.0);
  float norm=norm3(v0,v1,v2);
  float cosine=sum3(v0*seed[6],v1*seed[7],v2*seed[8])/maximum(norm*seed[9],1e-9f);
  float separation=norm3(c[0]-seed[3],c[1]-seed[4],c[2]-seed[5])-norm3(c[0]-seed[0],c[1]-seed[1],c[2]-seed[2]);
  float a=clip((cosine-cfg[0])/cfg[1]);
  float b=clip(gap/cfg[11]);
  float d=clip((separation+cfg[4])/cfg[5]);
  float reliability=clip(1.0f-noise[i]/cfg[6]);
  float value=((0.4f*a+0.35f*b)+0.25f*d)*square_root(reliability);
  bool identity=cosine>cfg[0] && norm>cfg[7];
  bool w=identity && (separation>-cfg[4] || gradient[i]>=cfg[8]) && value>=cfg[9];
  bool st=w && value>=cfg[10];
  score[i]=value;strong[i]=st;weak[i]=w && !st;uncertain[i]=identity && !w;
 }
}

extern "C" __declspec(dllexport) void color_predicate(
 const uint8_t* bgr,int size,const float* seed,const float* cfg,uint8_t* result) {
 const float bgblue=seed[5]-0.5f*(seed[3]+seed[4]);
 for(int i=0;i<size;++i) {
  result[i]=0;
  float r=bgr[3*i+2],g=bgr[3*i+1],b=bgr[3*i];
  float total=(r+g)+b,denom=total+1e-6f;
  float c0=r/denom,c1=g/denom,c2=b/denom,noise=3.0f/(total+3.0f);
  float blue=c2-0.5f*(c0+c1);
  if(!(cfg[0]-noise>0 && (blue-bgblue)-cfg[1]>0)) continue;
  float v0=c0-float(1.0/3.0),v1=c1-float(1.0/3.0),v2=c2-float(1.0/3.0);
  float norm=norm3(v0,v1,v2);
  float cosine=sum3(v0*seed[6],v1*seed[7],v2*seed[8])/maximum(norm*seed[9],1e-9f);
  float separation=(norm3(c0-seed[3],c1-seed[4],c2-seed[5])-norm3(c0-seed[0],c1-seed[1],c2-seed[2]))-cfg[3];
  result[i]=(norm-noise>0 && cosine-cfg[2]>0 && separation>0);
 }
}

extern "C" __declspec(dllexport) void project(
 int height,int width,int y0,int x0,const double* p,double* t,double* v) {
 for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
  int i=y*width+x;double dx=double(x+x0)-p[0],dy=double(y+y0)-p[1];
  t[i]=dx*p[2]+dy*p[3];v[i]=dx*p[4]+dy*p[5];
 }
}

extern "C" __declspec(dllexport) void domain(
 int height,int width,int y0,int x0,const double* p,const double* bounds,uint8_t* mask) {
 for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
  int i=y*width+x;double dx=double(x+x0)-p[0],dy=double(y+y0)-p[1];
  double t=dx*p[2]+dy*p[3],v=dx*p[4]+dy*p[5];
  mask[i]=(t>=bounds[0] && t<=bounds[1] && absolute(v-bounds[2])<=bounds[3]);
 }
}

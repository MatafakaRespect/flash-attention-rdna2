// fa_forward_gate.inc — standalone гейт (fp64-оракул) + бенч. Включается при -DFA_STANDALONE.
// Раскладка теста: Q,O = [SQ,NH,HD] (BSHD, batch=1); K,V = [NKV,SK,HD] (BHSD) — покрывает смешанные страйды.

static uint16_t f2bf16(float f){ uint32_t x; __builtin_memcpy(&x,&f,4);
    uint32_t r=x+0x7fffu+((x>>16)&1u); return (uint16_t)(r>>16); }
static float bf162f(uint16_t b){ uint32_t x=((uint32_t)b)<<16; float f; __builtin_memcpy(&f,&x,4); return f; }

// округление входа до dtype и обратно во float (для честного оракула)
static void round_dtype(std::vector<float>& v,int dt){
    for(auto& x:v){ if(dt==FA_F16) x=(float)(_Float16)x; else if(dt==FA_BF16) x=bf162f(f2bf16(x)); }
}
// упаковка float-массива в device-буфер нужного dtype
static void* to_dev(const std::vector<float>& v,int dt){
    void* d; size_t n=v.size();
    if(dt==FA_F32){ HIP_CHECK(hipMalloc(&d,n*4)); HIP_CHECK(hipMemcpy(d,v.data(),n*4,hipMemcpyHostToDevice)); return d; }
    if(dt==FA_F16){ std::vector<_Float16> h(n); for(size_t i=0;i<n;i++) h[i]=(_Float16)v[i];
        HIP_CHECK(hipMalloc(&d,n*2)); HIP_CHECK(hipMemcpy(d,h.data(),n*2,hipMemcpyHostToDevice)); return d; }
    std::vector<uint16_t> h(n); for(size_t i=0;i<n;i++) h[i]=f2bf16(v[i]);
    HIP_CHECK(hipMalloc(&d,n*2)); HIP_CHECK(hipMemcpy(d,h.data(),n*2,hipMemcpyHostToDevice)); return d;
}
static void from_dev(std::vector<float>& out,void* d,int dt){
    size_t n=out.size();
    if(dt==FA_F32){ HIP_CHECK(hipMemcpy(out.data(),d,n*4,hipMemcpyDeviceToHost)); return; }
    if(dt==FA_F16){ std::vector<_Float16> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*2,hipMemcpyDeviceToHost));
        for(size_t i=0;i<n;i++) out[i]=(float)h[i]; return; }
    std::vector<uint16_t> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*2,hipMemcpyDeviceToHost));
    for(size_t i=0;i<n;i++) out[i]=bf162f(h[i]);
}

// wl/wr — окно в семантике flash-attn: разрешено [qq+delta-wl .. qq+delta+wr], <0 = границы нет
struct TestSpec{ const char* name; int SQ,SK,HD,NH,gqa; bool causal,alibi; int wl,wr; float softcap; int chunk; };

static int run_test(const TestSpec& ts,int dt,double tol){
    int NKV=ts.NH/ts.gqa; float scale=1.0f/sqrtf((float)ts.HD);
    uint32_t s=1234567u+ts.HD*7+ts.NH+ts.SQ+dt*13;
    std::vector<float> Q((size_t)ts.SQ*ts.NH*ts.HD), K((size_t)NKV*ts.SK*ts.HD), V(K.size()), O(Q.size(),0.0f);
    std::vector<float> LSE((size_t)ts.NH*ts.SQ,0.0f);
    for(auto& x:Q)x=frand(s); for(auto& x:K)x=frand(s); for(auto& x:V)x=frand(s);
    round_dtype(Q,dt); round_dtype(K,dt); round_dtype(V,dt);
    std::vector<float> slopes(ts.NH);
    for(int h=0;h<ts.NH;h++) slopes[h]=-powf(2.0f,-(8.0f*(h+1)/ts.NH));  // ALiBi-подобные наклоны

    // оракул
    std::vector<float> Oref(Q.size()), LSEref((size_t)ts.NH*ts.SQ);
    RefCfg rc{ts.SQ,ts.SK,ts.HD,ts.NH,ts.gqa,scale,ts.softcap,ts.wl,ts.wr,ts.causal,ts.alibi,slopes.data(),ts.chunk};
    ref_attention(Q.data(),K.data(),V.data(),Oref.data(),LSEref.data(),rc);

    void *dQ=to_dev(Q,dt),*dK=to_dev(K,dt),*dV=to_dev(V,dt),*dO=to_dev(O,dt); float* dLSE;
    HIP_CHECK(hipMalloc(&dLSE,LSE.size()*4));
    FwdParams p{}; p.Q=dQ;p.K=dK;p.V=dV;p.O=dO;p.LSE=dLSE;
    p.sQ={(long long)ts.SQ*ts.NH*ts.HD,(long long)ts.HD,(long long)ts.NH*ts.HD};
    p.sK={(long long)NKV*ts.SK*ts.HD,(long long)ts.SK*ts.HD,(long long)ts.HD};
    p.sV=p.sK; p.sO=p.sQ;
    p.SQ=ts.SQ;p.SK=ts.SK;p.HD=ts.HD;p.NH=ts.NH;p.gqa=ts.gqa;p.scale=scale;
    p.softcap=ts.softcap;p.win_left=ts.wl;p.win_right=ts.wr;p.alibi_slopes=ts.alibi?nullptr:nullptr;
    p.causal=ts.causal;p.dropout_p=0.0f;p.philox_seed=0;p.cu_q=nullptr;p.cu_k=nullptr;p.batch=1;
    p.attn_chunk=ts.chunk;
    float* dSlopes=nullptr;
    if(ts.alibi){ HIP_CHECK(hipMalloc(&dSlopes,ts.NH*4)); HIP_CHECK(hipMemcpy(dSlopes,slopes.data(),ts.NH*4,hipMemcpyHostToDevice)); p.alibi_slopes=dSlopes; }
    fa_forward_launch(p,dt,0);
    HIP_CHECK(hipDeviceSynchronize());
    from_dev(O,dO,dt);
    HIP_CHECK(hipMemcpy(LSE.data(),dLSE,LSE.size()*4,hipMemcpyDeviceToHost));
    double rel=rel_rms(O.data(),Oref.data(),O.size());
    // LSE. У полностью замаскированной строки эталон = -inf, и ядро обязано дать то же.
    // Проверять через std::isinf нельзя: гейт собран с -ffast-math, там isinf врёт, а inf-inf
    // даёт nan — из-за этого пустые строки раньше просто не проверялись. Сравниваем по порогу.
    double le=0,lr=0; int bad_inf=0;
    for(size_t i=0;i<LSE.size();i++){
        bool ref_empty = !(LSEref[i] > -1e30f), got_empty = !(LSE[i] > -1e30f);
        if(ref_empty || got_empty){ if(ref_empty!=got_empty) bad_inf++; continue; }
        double d=LSE[i]-LSEref[i]; le+=d*d; lr+=LSEref[i]*LSEref[i]; }
    double lserel=lr>0?sqrt(le/lr):sqrt(le);
    const char* dn=dt==FA_F16?"fp16":dt==FA_BF16?"bf16":"fp32";
    bool ok=rel<tol && bad_inf==0;
    if(bad_inf) printf("      пустых строк разошлось: %d\n",bad_inf);
    printf("  [%s/%-4s] %-26s rel_rms=%.2e  LSE_rel=%.2e  %s\n",dn,ts.name,"",rel,lserel,ok?"OK":"FAIL");
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);hipFree(dLSE); if(dSlopes)hipFree(dSlopes);
    return ok?0:1;
}

// ---- varlen ----
// used>0 => проверяем seqused_q/seqused_k: в буфере лежат последовательности длиной seqs[],
// но реально используются только первые used_q[i] запросов и used_k[i] ключей.
static int run_varlen(int dt,double tol,bool seqused=false){
    int HD=128,NH=8,gqa=4,NKV=NH/gqa; float scale=1.0f/sqrtf((float)HD);
    int seqs[3]={100,64,200}; int B=3; int cq[4]={0,0,0,0}; for(int i=0;i<B;i++)cq[i+1]=cq[i]+seqs[i];
    int usedq[3]={70,64,150}, usedk[3]={90,30,180};
    int TOT=cq[B]; uint32_t s=999u+dt;
    std::vector<float> Q((size_t)TOT*NH*HD), K((size_t)NKV*TOT*HD), V(K.size()), O(Q.size(),0);
    for(auto&x:Q)x=frand(s); for(auto&x:K)x=frand(s); for(auto&x:V)x=frand(s);
    round_dtype(Q,dt);round_dtype(K,dt);round_dtype(V,dt);
    // оракул по каждому seq независимо
    std::vector<float> Oref(Q.size());
    for(int b=0;b<B;b++){ int off=cq[b];
        int LQ = seqused? usedq[b] : seqs[b];
        int LK = seqused? usedk[b] : seqs[b];
        // локальные представления через страйды: Q[(off+i)*NH+h]... но оракул ждёт compact — соберём срез
        std::vector<float> q((size_t)LQ*NH*HD),k((size_t)NKV*LK*HD),v(k.size()),o(q.size());
        for(int i=0;i<LQ;i++)for(int h=0;h<NH;h++)for(int d=0;d<HD;d++) q[(i*NH+h)*HD+d]=Q[((size_t)(off+i)*NH+h)*HD+d];
        for(int kh=0;kh<NKV;kh++)for(int j=0;j<LK;j++)for(int d=0;d<HD;d++){ k[(kh*LK+j)*HD+d]=K[((size_t)kh*TOT+off+j)*HD+d]; v[(kh*LK+j)*HD+d]=V[((size_t)kh*TOT+off+j)*HD+d]; }
        RefCfg rc{LQ,LK,HD,NH,gqa,scale,0,-1,-1,true,false,nullptr};
        ref_attention(q.data(),k.data(),v.data(),o.data(),nullptr,rc);
        // строки за пределами seqused_q ядро не пишет — в Oref они остаются нулями, как и в O
        for(int i=0;i<LQ;i++)for(int h=0;h<NH;h++)for(int d=0;d<HD;d++) Oref[((size_t)(off+i)*NH+h)*HD+d]=o[(i*NH+h)*HD+d];
    }
    void *dQ=to_dev(Q,dt),*dK=to_dev(K,dt),*dV=to_dev(V,dt),*dO=to_dev(O,dt);
    int *dcq; HIP_CHECK(hipMalloc(&dcq,(B+1)*4)); HIP_CHECK(hipMemcpy(dcq,cq,(B+1)*4,hipMemcpyHostToDevice));
    FwdParams p{}; p.Q=dQ;p.K=dK;p.V=dV;p.O=dO;p.LSE=nullptr;
    p.sQ={0,(long long)HD,(long long)NH*HD}; p.sK={0,(long long)TOT*HD,(long long)HD}; p.sV=p.sK; p.sO=p.sQ;
    p.SQ=0;p.SK=0;p.HD=HD;p.NH=NH;p.gqa=gqa;p.scale=scale;p.softcap=0;p.win_left=-1;p.win_right=-1;p.alibi_slopes=nullptr;
    p.causal=1;p.dropout_p=0;p.philox_seed=0;p.cu_q=dcq;p.cu_k=dcq;p.batch=B;
    int *duq=nullptr,*duk=nullptr;
    if(seqused){ HIP_CHECK(hipMalloc(&duq,B*4)); HIP_CHECK(hipMemcpy(duq,usedq,B*4,hipMemcpyHostToDevice));
                 HIP_CHECK(hipMalloc(&duk,B*4)); HIP_CHECK(hipMemcpy(duk,usedk,B*4,hipMemcpyHostToDevice));
                 p.seqused_q=duq; p.seqused_k=duk; }
    // grid.x должен покрыть макс. seq
    int maxL=0; for(int i=0;i<B;i++) maxL=std::max(maxL,seqs[i]);
    // launch напрямую (SQ в диспетчере = grid.x); подменим через временный SQ=maxL для сетки
    p.SQ=maxL;
    fa_forward_launch(p,dt,0);
    HIP_CHECK(hipDeviceSynchronize());
    from_dev(O,dO,dt);
    double rel=rel_rms(O.data(),Oref.data(),O.size());
    if(seqused && rel>=tol){     // разбор по последовательностям — только когда гейт красный
        for(int b=0;b<B;b++){ int off=cq[b];
            double r1=rel_rms(O.data()+(size_t)off*NH*HD, Oref.data()+(size_t)off*NH*HD,(size_t)usedq[b]*NH*HD);
            double r2=rel_rms(O.data()+(size_t)(off+usedq[b])*NH*HD, Oref.data()+(size_t)(off+usedq[b])*NH*HD,
                              (size_t)(seqs[b]-usedq[b])*NH*HD);
            printf("      seq%d used_q=%d used_k=%d: строки в пределах used %.2e, хвост %.2e\n",b,usedq[b],usedk[b],r1,r2); }
    }
    const char* dn=dt==FA_F16?"fp16":dt==FA_BF16?"bf16":"fp32";
    printf("  [%s/varlen%s] 3 seq (100,64,200) HD128 GQA4  rel_rms=%.2e  %s\n",
           dn,seqused?"+seqused":"        ",rel,rel<tol?"OK":"FAIL");
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);hipFree(dcq);
    if(duq){hipFree(duq);hipFree(duk);}
    return rel<tol?0:1;
}

static void bench(int dt,int S,int window){
    int NH=32,gqa=4,NKV=NH/gqa,HD=128; float scale=1.0f/sqrtf((float)HD);
    std::vector<float> Q((size_t)S*NH*HD,0.02f),K((size_t)NKV*S*HD,0.02f),V(K.size(),0.02f),O(Q.size());
    void*dQ=to_dev(Q,dt),*dK=to_dev(K,dt),*dV=to_dev(V,dt),*dO=to_dev(O,dt);
    FwdParams p{}; p.Q=dQ;p.K=dK;p.V=dV;p.O=dO;p.LSE=nullptr;
    p.sQ={0,(long long)HD,(long long)NH*HD}; p.sK={0,(long long)S*HD,(long long)HD}; p.sV=p.sK; p.sO=p.sQ;
    p.SQ=S;p.SK=S;p.HD=HD;p.NH=NH;p.gqa=gqa;p.scale=scale;p.softcap=0;
    p.win_left=(window<0?-1:window-1);p.win_right=(window<0?-1:0);p.alibi_slopes=nullptr;
    p.causal=1;p.dropout_p=0;p.philox_seed=0;p.cu_q=nullptr;p.cu_k=nullptr;p.batch=1;
    fa_forward_launch(p,dt,0); HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0,t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    int iters = S>=8192?10:20;
    HIP_CHECK(hipEventRecord(t0));
    for(int i=0;i<iters;i++) fa_forward_launch(p,dt,0);
    HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
    float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,t0,t1)); ms/=iters;
    // FLOPs: полный причинный треугольник, либо оконная полоса
    double pairs = window<0? (double)S*S/2 : (double)S*window - (double)window*window/2;
    double tf=2.0*NH*pairs*HD*2/(ms*1e-3)/1e12;
    double kvGB=(double)NKV*S*HD*2*2/1e9;  // K+V один проход (нижняя оценка трафика)
    const char* dn=dt==FA_F16?"fp16":"bf16";
    if(window<0) printf("  [%s] S=%5d HD128 GQA4 causal      : %.3f мс  ~%.1f TFLOP/s\n",dn,S,ms,tf);
    else         printf("  [%s] S=%5d HD128 GQA4 window=%-5d: %.3f мс  ~%.1f TFLOP/s (разреженный путь)\n",dn,S,window,ms,tf);
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
}

int main(){
    int fails=0;
    printf("=== fa_forward: гейт корректности (fp64-оракул) ===\n");
    // допуски: fp16 2.6e-4, bf16 4e-3, fp32 1e-5 (пол типа; bf16 свободнее)
    TestSpec base[]={
        {"d128 causal GQA4",   256,256,128,32,4, true,false,-1,-1,0.0f},
        {"d128 causal GQA1",   256,256,128, 8,1, true,false,-1,-1,0.0f},
        {"d128 full(noncausal)",192,192,128,16,4,false,false,-1,-1,0.0f},
        {"d128 SQ<SK bottomR", 128,320,128,16,4, true,false,-1,-1,0.0f},
        // SQ>SK при causal: первые SQ-SK строк не видят НИ ОДНОГО ключа. flash-attn отдаёт там
        // нули и LSE=-inf. Раньше это не проверялось, и ядро отдавало сумму V.
        {"d128 SQ>SK пустые строки",256,64,128,16,4,true,false,-1,-1,0.0f},
        // окна: левое (как раньше), двустороннее, только правое, ровно диагональ
        {"d128 окно (63,0)",   256,256,128,16,4, true,false,63, 0,0.0f},
        {"d128 окно (32,32)",  256,256,128,16,4,false,false,32,32,0.0f},
        {"d128 окно (0,0) диаг",256,256,128,16,4,false,false, 0, 0,0.0f},
        {"d128 окно (-1,0)=causal",256,256,128,16,4,false,false,-1,0,0.0f},
        {"d128 окно (0,-1)",   256,256,128,16,4,false,false, 0,-1,0.0f},
        {"d128 окно (16,48)",  192,192,128,16,4,false,false,16,48,0.0f},
        {"d128 окно (32,32) SQ<SK",128,320,128,16,4,false,false,32,32,0.0f},
        {"d128 softcap=30",    200,200,128,16,4, true,false,-1,-1,30.0f},
        {"d128 alibi",         200,200,128,16,4, true,true ,-1,-1,0.0f},
        {"d64",                200,200, 64,16,4, true,false,-1,-1,0.0f},
        {"d80",                200,200, 80,16,4, true,false,-1,-1,0.0f},
        {"d96",                160,160, 96,16,4, true,false,-1,-1,0.0f},
        {"d256",               160,160,256, 8,4, true,false,-1,-1,0.0f},
        {"d32",                160,160, 32,16,4, true,false,-1,-1,0.0f},
        // новые head_dim: инстанцируются конфигом BR=64,BC=32 (при BR=128 не влезают в LDS)
        {"d112",               160,160,112,16,4, true,false,-1,-1,0.0f},
        {"d160",               160,160,160, 8,4, true,false,-1,-1,0.0f},
        {"d192",               160,160,192, 8,4, true,false,-1,-1,0.0f},
        {"d224",               160,160,224, 8,4, true,false,-1,-1,0.0f},
        {"d192 full",          128,192,192, 8,2,false,false,-1,-1,0.0f},
        // attention_chunk (FA-3): запрос видит только ключи своего чанка.
        // Берём чанк и кратный тайлу (64), и НЕкратный (100) — чтобы поймать ошибку выравнивания.
        {"chunk=64 causal",    256,256,128,16,4, true,false,-1,-1,0.0f, 64},
        {"chunk=100 causal",   256,256,128,16,4, true,false,-1,-1,0.0f,100},
        {"chunk=128 full",     256,256,128,16,4,false,false,-1,-1,0.0f,128},
        {"chunk=64 SQ<SK",     128,320,128,16,4, true,false,-1,-1,0.0f, 64},
    };
    for(int dt=0;dt<3;dt++){ double tol = dt==FA_F16?3e-4:dt==FA_BF16?5e-3:1e-5;
        for(auto& ts:base) fails+=run_test(ts,dt,tol);
        double vtol = dt==FA_F16?3e-4:dt==FA_BF16?5e-3:1e-5;
        fails+=run_varlen(dt, vtol, false);
        fails+=run_varlen(dt, vtol, true);
    }
    printf("\n=== бенч (один короткий прогон на форму) ===\n");
    bench(FA_F16,2048,-1); bench(FA_BF16,2048,-1);
    bench(FA_F16,8192,-1); bench(FA_BF16,8192,-1);
    bench(FA_F16,8192,512);   // разреженный (sliding-window) — трафик и работа ∝ window, не S
    printf(fails?"\n*** ЕСТЬ ПАДЕНИЯ: %d ***\n":"\nВСЕ ГЕЙТЫ ЗЕЛЁНЫЕ\n",fails);
    return fails;
}

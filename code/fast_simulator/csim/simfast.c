/*
 * simfast.c -- C version of simulation_fast._run (rules of simulation_pso_paper_v2), for a speed comparison with the
 * Numba version. Same algorithm statement by statement, and the same random numbers: Numba's Mersenne Twister
 * (numba/_random.c: numba_rnd_init, get_next_double) with exponentials drawn as -log(1 - u) * scale, so for the same
 * seed the two versions must produce identical output.
 * Build: gcc -O3 -march=native -shared -fPIC -o libsimfast.so simfast.c -lm
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfU
#define MT_UPPER_MASK 0x80000000U
#define MT_LOWER_MASK 0x7fffffffU

typedef struct { int index; unsigned int mt[MT_N]; } rng_t;

static void rnd_init(rng_t *s, unsigned int seed) {
    for (unsigned int pos = 0; pos < MT_N; pos++) {
        s->mt[pos] = seed;
        seed = (1812433253U * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffU;
    }
    s->index = MT_N;
}

static void rnd_shuffle(rng_t *s) {
    int i; unsigned int y;
    for (i = 0; i < MT_N - MT_M; i++) {
        y = (s->mt[i] & MT_UPPER_MASK) | (s->mt[i + 1] & MT_LOWER_MASK);
        s->mt[i] = s->mt[i + MT_M] ^ (y >> 1) ^ (-(int)(y & 1) & MT_MATRIX_A);
    }
    for (; i < MT_N - 1; i++) {
        y = (s->mt[i] & MT_UPPER_MASK) | (s->mt[i + 1] & MT_LOWER_MASK);
        s->mt[i] = s->mt[i + (MT_M - MT_N)] ^ (y >> 1) ^ (-(int)(y & 1) & MT_MATRIX_A);
    }
    y = (s->mt[MT_N - 1] & MT_UPPER_MASK) | (s->mt[0] & MT_LOWER_MASK);
    s->mt[MT_N - 1] = s->mt[MT_M - 1] ^ (y >> 1) ^ (-(int)(y & 1) & MT_MATRIX_A);
}

static inline unsigned int next_int32(rng_t *s) {
    if (s->index >= MT_N) { rnd_shuffle(s); s->index = 0; }
    unsigned int y = s->mt[s->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

static inline double next_double(rng_t *s) {
    double a = next_int32(s) >> 5, b = next_int32(s) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

static inline double expo(rng_t *s, double scale) { return -log(1.0 - next_double(s)) * scale; }

#define IDX(k, i) ((long)(k) * cap + (i))

static inline void stat(int k, double t, const long *count, double *last, double *dist, double *over, double warm, int maxq) {
    if (t > warm) {
        double start = last[k] > warm ? last[k] : warm;
        long c = count[k];
        if (c < maxq) dist[(long)k * maxq + c] += t - start; else over[k] += t - start;
    }
    last[k] = t;
}

typedef struct {
    int N, M; long cap; const double *R; double *qa; signed char *qf; long *qhead, *qlen;
    long *busy_k; double *busy_a, *busy_since, *comp, *starts, *whist, *nowait; const double *buckets; int nb; rng_t *rng;
} ctx_t;

static inline void start_job(ctx_t *c, int j, int k, double t) {
    long cap = c->cap;
    long h = c->qhead[k];
    double a = c->qa[IDX(k, h)];
    signed char f = c->qf[IDX(k, h)];
    c->qhead[k] = (h + 1) % cap;
    c->qlen[k] -= 1;
    if (f == 0) {
        double w = t - a;
        if (w <= 0.0) c->nowait[k] += 1.0;
        for (int b = 0; b < c->nb; b++) if (w <= c->buckets[b]) c->whist[(long)k * c->nb + b] += 1.0;
    }
    c->busy_k[j] = k; c->busy_a[j] = a; c->busy_since[j] = t;
    c->comp[j] = t + expo(c->rng, 1.0 / c->R[(long)j * c->N + k]);
    c->starts[(long)j * c->N + k] += 1.0;
}

/* returns 1 if the queue capacity was exceeded */
int run(const double *lam, const double *R, const double *score, int N, int M, double n_requests, double warm_requests,
        unsigned int seed, int maxq, const double *buckets, int nb, long cap,
        double *dist, double *over, double *util, double *starts, double *arrivals, double *nowait, double *whist, int preempt) {
    rng_t rng; rnd_init(&rng, seed);
    double Lam = 0; for (int k = 0; k < N; k++) Lam += lam[k];
    double *cum = malloc(sizeof(double) * N); double acc = 0;
    for (int k = 0; k < N; k++) { acc += lam[k] / Lam; cum[k] = acc; }      /* np.cumsum(lam / Lam) */
    double stop = n_requests / Lam, warm = warm_requests / Lam;
    double *qa = malloc(sizeof(double) * N * cap); signed char *qf = calloc((size_t)N * cap, 1);
    long *qhead = calloc(N, sizeof(long)), *qlen = calloc(N, sizeof(long)), *count = calloc(N, sizeof(long));
    double *last = calloc(N, sizeof(double));
    long *busy_k = malloc(sizeof(long) * M); double *busy_a = calloc(M, sizeof(double)), *busy_since = calloc(M, sizeof(double));
    double *comp = malloc(sizeof(double) * M), *idle_since = calloc(M, sizeof(double));
    for (int j = 0; j < M; j++) { busy_k[j] = -1; comp[j] = INFINITY; }
    ctx_t c = {N, M, cap, R, qa, qf, qhead, qlen, busy_k, busy_a, busy_since, comp, starts, whist, nowait, buckets, nb, &rng};
    int overflow = 0;
    double t = 0.0, next_arr = expo(&rng, 1.0 / Lam);
    while (1) {
        int jmin = -1; double tc = INFINITY;
        for (int j = 0; j < M; j++) if (comp[j] < tc) { tc = comp[j]; jmin = j; }
        double t_next = next_arr <= tc ? next_arr : tc;
        if (t_next >= stop) break;
        t = t_next;
        if (next_arr <= tc) {                                              /* arrival */
            double u = next_double(&rng);
            int k = 0;
            while (k < N - 1 && cum[k] <= u) k++;
            stat(k, t, count, last, dist, over, warm, maxq);
            count[k] += 1; arrivals[k] += 1.0;
            if (qlen[k] >= cap) { overflow = 1; break; }
            long idx = (qhead[k] + qlen[k]) % cap;
            qa[IDX(k, idx)] = t; qf[IDX(k, idx)] = 0; qlen[k] += 1;
            int kk = k;
            while (kk >= 0) {                                              /* dispatch, with cascade */
                int jbest = -1; double best_idle = INFINITY;
                for (int j = 0; j < M; j++)
                    if (R[(long)j * N + kk] > 0.0 && busy_k[j] < 0 && idle_since[j] < best_idle) { best_idle = idle_since[j]; jbest = j; }
                if (jbest >= 0) { start_job(&c, jbest, kk, t); break; }
                if (!preempt) break;                                        /* non-preemptive priorities */
                int jv = -1; double vs = -INFINITY, va = -INFINITY;
                for (int j = 0; j < M; j++) {
                    if (R[(long)j * N + kk] > 0.0 && busy_k[j] >= 0) {
                        double cs = score[(long)j * N + busy_k[j]];
                        if (score[(long)j * N + kk] < cs && (cs > vs || (cs == vs && busy_a[j] > va))) { jv = j; vs = cs; va = busy_a[j]; }
                    }
                }
                if (jv < 0) break;
                int v = (int)busy_k[jv];
                util[(long)jv * N + v] += t - busy_since[jv];
                long h = ((qhead[v] - 1) % cap + cap) % cap;               /* Python's (x - 1) % cap */
                qhead[v] = h; qa[IDX(v, h)] = busy_a[jv]; qf[IDX(v, h)] = 1; qlen[v] += 1;
                busy_k[jv] = -1; comp[jv] = INFINITY;
                start_job(&c, jv, kk, t);
                kk = v;
            }
            next_arr = t + expo(&rng, 1.0 / Lam);
        } else {                                                           /* completion */
            int j = jmin; int k = (int)busy_k[j];
            util[(long)j * N + k] += t - busy_since[j];
            stat(k, t, count, last, dist, over, warm, maxq);
            count[k] -= 1; busy_k[j] = -1; comp[j] = INFINITY; idle_since[j] = t;
            int best = -1; double bs = INFINITY, ba = INFINITY;
            for (int kk = 0; kk < N; kk++) {
                if (R[(long)j * N + kk] > 0.0 && qlen[kk] > 0) {
                    double s = score[(long)j * N + kk], a = qa[IDX(kk, qhead[kk])];
                    if (s < bs || (s == bs && a < ba)) { best = kk; bs = s; ba = a; }
                }
            }
            if (best >= 0) start_job(&c, j, best, t);
        }
    }
    for (int k = 0; k < N; k++) stat(k, stop, count, last, dist, over, warm, maxq);   /* flush */
    for (int j = 0; j < M; j++) if (busy_k[j] >= 0) util[(long)j * N + busy_k[j]] += stop - busy_since[j];
    double span = stop - warm;
    for (long i = 0; i < (long)N * maxq; i++) dist[i] /= span;
    for (int k = 0; k < N; k++) over[k] /= span;
    for (long i = 0; i < (long)M * N; i++) util[i] /= stop;
    free(cum); free(qa); free(qf); free(qhead); free(qlen); free(count); free(last);
    free(busy_k); free(busy_a); free(busy_since); free(comp); free(idle_since);
    return overflow;
}

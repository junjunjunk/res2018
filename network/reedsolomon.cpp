/**
 *  Reed-Solomon符号
 *  GF(2^8)=GF(256) GF(2^4)=GF(16)に対応 ガロア体
 *　デフォルトGF(2^8) T=8
 *  参考　11.Oct ,2013 t-matsu ReedSolomon.java
 */

#include <math.h>
#include <array>
#include <cstddef>
#include <iostream>
#include <vector>
#define ARRAY_LENGTH(array) (sizeof(array) / sizeof(array[0]))
using namespace std;

typedef unsigned char byte;

byte test[3] = {0, 127, 255};  // unsigned char

static int len_enc;
static int len_ans;

/**
 *  Reed-Solomon符号
 *  GF(2^8) GF(2^4)に対応
 *　デフォルトGF(2^8) T=8
 *  11.Oct ,2013 t-matsu
 */

class ReedSolomon {
  int M = 8;    // GF(2^M) 1シンボル 8ビット
  int N = 255;  // 2^M -1 各ブロックの総シンボル数
  int T = 8;  // N-K = 2T　訂正能力　2Tは各ブロックに加えるパリティ・シンボル数
  int K = 239;  // data　各ブロックの情報シンボル数

  int *alpha_to;
  int *index_of;
  int *G;
  int P8[9] = {1, 0, 1, 1, 1, 0, 0, 0, 1};
  int P4[5] = {1, 1, 0, 0, 1};

  /**
     m: GF(2^m)のM指定
     n: 冗長データ＋データ つまりエンコード後の要素数
     n-kの冗長データを追加している t: 誤り訂正能力Tの指定　のコンストラクタ k:
     冗長データを除いたデータ
   */
 public:
  ReedSolomon(int m, int n, int t, int k) {
    if ((m == 8 || m == 4)) {
      M = m;
      T = t;
      N = (int)(pow(2, M)) - 1;
      K = N - (2 * T);
      init();
    } else {
      perror("M != (4 or 8)");
      exit(0);
    }
  }

  /**
     m: GF(2^m)のM指定
     t: 誤り訂正能力Tの指定　のコンストラクタ
   */
 public:
  ReedSolomon(int m, int t) {
    if ((m == 8 || m == 4)) {
      M = m;
      T = t;
      N = (int)(pow(2, M)) - 1;
      K = N - (2 * T);
      init();
    } else {
      perror("M != (4 or 8)");
      exit(0);
    }
  }

  /**
     N=255固定 誤り訂正能力Tを設定するコンストラクタ
  */
 public:
  ReedSolomon(int t) {
    T = t;
    K = N - (2 * T);
    init();
  }

  /**
     N=255固定　誤り訂正能力T=8 固定のデフォルトコンストラクタ
  */
 public:
  ReedSolomon() { init(); }

  void init() {
    G = new int[N - K + 1];
    alpha_to = new int[N + 1];
    index_of = new int[N + 1];
    int *P = (M == 8) ? P8 : P4;
    int i, j, mask;
    mask = 1;

    alpha_to[M] = 0;
    for (i = 0; i < M; i++) {
      alpha_to[i] = mask;
      index_of[alpha_to[i]] = i;
      if (P[i] != 0) alpha_to[M] ^= mask;
      mask <<= 1;
    }
    index_of[alpha_to[M]] = M;
    mask >>= 1;
    for (i = M + 1; i < N; i++) {
      if (alpha_to[i - 1] >= mask)
        alpha_to[i] = alpha_to[M] ^ ((alpha_to[i - 1] ^ mask) << 1);
      else
        alpha_to[i] = alpha_to[i - 1] << 1;
      index_of[alpha_to[i]] = i;
    }
    index_of[0] = -1;

    G[0] = 2;
    G[1] = 1;
    for (i = 2; i <= N - K; i++) {
      G[i] = 1;
      for (j = i - 1; j > 0; j--)
        if (G[j] != 0)
          G[j] = G[j - 1] ^ alpha_to[(index_of[G[j]] + i) % N];
        else
          G[j] = G[j - 1];
      G[0] = alpha_to[(index_of[G[0]] + i) % N];
    }
    for (i = 0; i <= N - K; i++) G[i] = index_of[G[i]];
    printf("M=%d\nN=%d\nT=%d\nK=%d\n", M, N, T, K);
  }

 public:
  int getM() { return M; }

 public:
  int getN() { return N; }

 public:
  int getK() { return K; }

 public:
  int getT() { return T; }

 public:
  static byte *combine4(int len, byte d[]) {
    byte *tmp = new byte[len / 2 + len % 2];
    for (int i = 0; i < ARRAY_LENGTH(tmp); i++) {
      if (i * 2 + 1 < len)
        tmp[i] = (byte)(d[i * 2 + 1] & 0x0F + ((d[i * 2] & 0x0F) << 4));
      else
        tmp[i] = (byte)((d[i * 2] & 0x0F) << 4);
    }
    return tmp;
  }

 public:
  static byte *div4(int len, byte d[]) {
    byte *tmp = new byte[len * 2];
    for (int i = 0; i < len; i++) {
      tmp[i * 2 + 1] = (byte)(d[i] & 0x0F);
      tmp[i * 2] = (byte)((d[i] & 0xFF) >> 4);
    }
    return tmp;
  }

  /**
     データ部分のエンコードを行う
     入力byte[]に対して入力byte[].length/Kごとのブロックを生成し，
     各ブロックのindexごとに冗長ヘッダを作成する
     実際のエンコード(と冗長データ作成)はencode_rs()
  */
 public:
  byte *encode(int len, byte d[]) {
    int turn_len = len / K;
    printf("method:encode len=%d,K=%d,turn_len=%d\n", len, K, turn_len);
    if (len % K != 0) turn_len++;
    byte *enc = new byte[N * turn_len];
    len_enc = N * turn_len;

    printf("method:encode enc_length=%d\n", N * turn_len);
    int *data = new int[K];

    for (int i = 0; i < turn_len; i++) {
      for (int j = 0; j < K; j++) {
        data[j] = (j * turn_len + i < len) ? (d[i + turn_len * j] & 0xFF) : 0;
        // printf("data[%d]=%d\n",j,data[j]);
        enc[(N - K) * turn_len + j + K * i] =
            (i * K + j < len) ? d[i * K + j] : 0;
        // printf("enc[%d]=%d\n",(N-K)*turn_len+j+K*i,enc[(N-K)*turn_len+j+K*i]);
      }

      int *B = encode_rs(K, data);
      for (int j = 0; j < N - K; j++) {
        enc[i + turn_len * j] = (byte)B[j];
        // printf("enc[%d]=%d\n",j,i+turn_len*j);
        // printf("i+turn_len*j=%d,j=%d\n",i+turn_len*j,j);
      }
    }
    printf("<   encode end   >\n");
    return enc;
  }

  /**
     Kシンボルのエンコードを行う
     入力int data[K]に対して生成した冗長データのint[N-K]を返す
  */

 public:
  int *encode_rs(int len, int data[]) {
    int i, j;
    int feedback;
    int *B = new int[N - K];
    printf("len=%d K=%d\n", len, K);
    if (len != K) {
      printf("encode size error\n len=%d K=%d", len, K);
      return nullptr;
    }
    for (i = 0; i < N - K; i++) B[i] = 0;
    for (i = K - 1; i >= 0; i--) {
      feedback = index_of[data[i] ^ B[N - K - 1]];
      if (feedback != -1) {
        for (j = N - K - 1; j > 0; j--)
          if (G[j] != -1)
            B[j] = B[j - 1] ^ alpha_to[(G[j] + feedback) % N];
          else
            B[j] = B[j - 1];
        B[0] = alpha_to[(G[0] + feedback) % N];
      } else {
        for (j = N - K - 1; j > 0; j--) B[j] = B[j - 1];
        B[0] = 0;
      }
    }
    return B;
  }

  /**
     デコードしてデータを並び替えて取り出す
     実際のデコードはdecode_rs()
  */
 public:
  byte *decode(int len, byte d[]) {
    int turn_len = len / N;
    if (len % N != 0) turn_len++;  //たぶん実行されないはず
    byte *ans = new byte[K * turn_len];
    len_ans = K * turn_len;
    int *recd = new int[N];

    for (int i = 0; i < turn_len; i++) {
      for (int j = 0; j < N - K; j++)
        recd[j] = index_of[d[i + turn_len * j] & 0xFF];
      for (int j = 0; j < K; j++)
        recd[j + N - K] = index_of[d[i + turn_len * (N - K + j)] & 0xFF];
      recd = decode_rs(N, recd);
      for (int j = 0; j < K; j++) ans[i + turn_len * j] = (byte)recd[j + N - K];
    }
    return ans;
  }

  /**
     デコードを行う Nシンボル→Nシンボル
     冗長データの除去は別途行う必要がある
  */
 public:
  int *decode_rs(int len, int recd[]) {
    int i, j, u, q;
    // int *elp = new int[N-K+2][N-K];
    vector<vector<int> > elp;
    elp.resize(N - K + 2);
    for (int i = 0; i < N - K + 2; i++) {
      elp[i].resize(N - K);
    }

    int *d = new int[N - K + 2];
    int *l = new int[N - K + 2];
    int *u_lu = new int[N - K + 2];
    int *s = new int[N - K + 1];
    int count = 0, syn_error = 0;
    int *root = new int[T];
    int *loc = new int[T];
    int *z = new int[T + 1];
    int *err = new int[N];
    int *reg = new int[T + 1];

    if (len != N) {
      perror("decode size error");
      return nullptr;
    }
    /* first form the syndromes */
    for (i = 1; i <= N - K; i++) {
      s[i] = 0;
      for (j = 0; j < N; j++)
        if (recd[j] != -1) s[i] ^= alpha_to[(recd[j] + i * j) % N];
      if (s[i] != 0) syn_error = 1;
      s[i] = index_of[s[i]];
    }

    if (syn_error > 0) { /* エラーがあった場合訂正を試みる */
      d[0] = 0;
      d[1] = s[1];
      elp[0][0] = 0;
      elp[1][0] = 1;
      for (i = 1; i < N - K; i++) {
        elp[0][i] = -1;
        elp[1][i] = 0;
      }
      l[0] = 0;
      l[1] = 0;
      u_lu[0] = -1;
      u_lu[1] = 0;
      u = 0;

      do {
        u++;
        if (d[u] == -1) {
          l[u + 1] = l[u];
          for (i = 0; i <= l[u]; i++) {
            elp[u + 1][i] = elp[u][i];
            elp[u][i] = index_of[elp[u][i]];
          }
        } else {
          q = u - 1;
          while ((d[q] == -1) && (q > 0)) q--;
          if (q > 0) {
            j = q;
            do {
              j--;
              if ((d[j] != -1) && (u_lu[q] < u_lu[j])) q = j;
            } while (j > 0);
          }
          if (l[u] > l[q] + u - q)
            l[u + 1] = l[u];
          else
            l[u + 1] = l[q] + u - q;

          for (i = 0; i < N - K; i++) elp[u + 1][i] = 0;
          for (i = 0; i <= l[q]; i++)
            if (elp[q][i] != -1)
              elp[u + 1][i + u - q] =
                  alpha_to[(d[u] + N - d[q] + elp[q][i]) % N];
          for (i = 0; i <= l[u]; i++) {
            elp[u + 1][i] ^= elp[u][i];
            elp[u][i] = index_of[elp[u][i]];
          }
        }
        u_lu[u + 1] = u - l[u + 1];

        if (u < N - K) {
          if (s[u + 1] != -1)
            d[u + 1] = alpha_to[s[u + 1]];
          else
            d[u + 1] = 0;
          for (i = 1; i <= l[u + 1]; i++)
            if ((s[u + 1 - i] != -1) && (elp[u + 1][i] != 0))
              d[u + 1] ^=
                  alpha_to[(s[u + 1 - i] + index_of[elp[u + 1][i]]) % N];
          d[u + 1] = index_of[d[u + 1]];
        }
      } while ((u < N - K) && (l[u + 1] <= T));

      u++;
      if (l[u] <= T) {
        for (i = 0; i <= l[u]; i++) elp[u][i] = index_of[elp[u][i]];

        for (i = 1; i <= l[u]; i++) reg[i] = elp[u][i];
        count = 0;
        for (i = 1; i <= N; i++) {
          q = 1;
          for (j = 1; j <= l[u]; j++)
            if (reg[j] != -1) {
              reg[j] = (reg[j] + j) % N;
              q ^= alpha_to[reg[j]];
            }
          if (!(q > 0)) {
            root[count] = i;
            loc[count] = N - i;
            count++;
          }
        }
        if (count == l[u]) {
          for (i = 1; i <= l[u]; i++) {
            if ((s[i] != -1) && (elp[u][i] != -1))
              z[i] = alpha_to[s[i]] ^ alpha_to[elp[u][i]];
            else if ((s[i] != -1) && (elp[u][i] == -1))
              z[i] = alpha_to[s[i]];
            else if ((s[i] == -1) && (elp[u][i] != -1))
              z[i] = alpha_to[elp[u][i]];
            else
              z[i] = 0;
            for (j = 1; j < i; j++)
              if ((s[j] != -1) && (elp[u][i - j] != -1))
                z[i] ^= alpha_to[(elp[u][i - j] + s[j]) % N];
            z[i] = index_of[z[i]];
          }

          for (i = 0; i < N; i++) {
            err[i] = 0;
            if (recd[i] != -1)
              recd[i] = alpha_to[recd[i]];
            else
              recd[i] = 0;
          }
          for (i = 0; i < l[u]; i++) {
            err[loc[i]] = 1;
            for (j = 1; j <= l[u]; j++)
              if (z[j] != -1) err[loc[i]] ^= alpha_to[(z[j] + j * root[i]) % N];
            if (err[loc[i]] != 0) {
              err[loc[i]] = index_of[err[loc[i]]];
              q = 0;
              for (j = 0; j < l[u]; j++)
                if (j != i) q += index_of[1 ^ alpha_to[(loc[j] + root[i]) % N]];
              q = q % N;
              err[loc[i]] = alpha_to[(err[loc[i]] - q + N) % N];
              recd[loc[i]] ^= err[loc[i]];
            }
          }
        } else
          for (i = 0; i < N; i++)
            if (recd[i] != -1)
              recd[i] = alpha_to[recd[i]];
            else
              recd[i] = 0;
      } else
        for (i = 0; i < N; i++)
          if (recd[i] != -1)
            recd[i] = alpha_to[recd[i]];
          else
            recd[i] = 0;
    } else
      for (i = 0; i < N; i++)
        if (recd[i] != -1)
          recd[i] = alpha_to[recd[i]];
        else
          recd[i] = 0;
    return recd;
  }
};

/**
   デバッグ用main()
*/
int main(int argc, char *args[]) {
  //    ReedSolomon *rs = new ReedSolomon();

  ReedSolomon *rs = new ReedSolomon(8, 2);  // M=8 T=8 rs(255,239)
  // ReedSolomon *rs = new ReedSolomon(4,2); //M=4 T=2 rs(15,11)
  // ReedSolomon *rs = new ReedSolomon(8,20,2,16); //M=4 T=2 rs(15,11)
  if (argc != 2) {
    printf("need 1 argument\n");
    return 1;
  }
  int size = atoi(args[1]);
  if (size < 1) {
    printf("error : size <1 :%d\n", size);
    return 1;
  }
  /* make sample message */
  byte *mes = new byte[size];
  for (int i = 0; i < size; i++) {  // mod256
    mes[i] = (byte)(i % (rs->getN() + 1));
  }
  if (rs->getM() == 4) mes = rs->div4(size, mes);

  /* encode */
  printf("<   start encode   >\n");
  byte *enc = rs->encode(size, mes);
  printf("len_enc=%d\n", len_enc);
  // for(int i=0;i<len_enc;i++)printf("%d\n",int(enc[i]));

  /* insert error */
  for (int i = 0; i < 50; i++) {
    printf("i=%d correct %d\n", len_enc / 2 + i, enc[len_enc / 2 + i]);
    enc[len_enc / 5 + i] = 0;
    printf("mis %d\n", enc[len_enc / 2 + i]);
  }

  /* decode */
  byte *ans = rs->decode(len_enc, enc);

  // System.out.printf("  i mes enc ans\n");
  int error_num = 0;
  for (int i = 0; i < len_enc; i++) {
    int m = (i < size) ? (mes[i] & 0xFF) : 0;
    int e = (i < len_enc) ? (enc[i] & 0xFF) : 0;
    int a = (i < len_ans) ? (ans[i] & 0xFF) : 0;
    printf("%3d %3d %3d\n", m, e, a);
    // System.out.printf("%3d %3d %3d %3d\n",i,m,e,a);
    if (m != a) error_num++;
  }
  printf("error: %d/%d\n", error_num, size);

  delete rs;

  /*
  for(int i=0;i<ARRAY_LENGTH(mes);i++)
      System.out.printf("i:%3d mes:%3d enc:%3d
  dec:%3d\n",i,(mes[i]&0xFF),enc[i]&0xFF,(ans[i]& 0xFF));
  */
  return 0;
}

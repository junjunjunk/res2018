#include <arpa/inet.h>
#include <math.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#define ARRAY_LENGTH(array) (sizeof(array) / sizeof(array[0]))

static const float ROWS = 224;
static const float COLS = 224;
static const int PORT = 80;

static const int DATA_SIZE = 700;
static const int HEAD_SIZE = 10;
// static const int SEND_TIME = 5 * 1000000; //μs

static const int MAX_DEVICE = 2;
static unsigned char imgcode[216][DATA_SIZE];
static unsigned char tcode[DATA_SIZE][255];
static unsigned char sendcode[255][DATA_SIZE];

typedef unsigned char byte;
static int len_enc;

typedef struct {
  char name[100];
  int no;
  int fd;

  char buff[HEAD_SIZE + DATA_SIZE];
  struct sockaddr_in addr;
  socklen_t addrSize;
  char feedback = 0;
} Mem;

Mem member[MAX_DEVICE];
cv::Mat imgs[MAX_DEVICE];

static int broadcastImg(char file_name[]);
static int openTcp();
static void setSeq(int num, char *buff);
static int checkSeq(char *buff);
void receiveImg(Mem mem);
void feedback(Mem mem);

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
 n: 冗長データ＋データ つまりエンコード後の要素数 n-kの冗長データを追加している
 t: 誤り訂正能力Tの指定　のコンストラクタ
 k: 冗長データを除いたデータ
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

  /*
         データ部分のエンコードを行う
         入力byte[]に対して入力byte[].length/Kごとのブロックを生成し，
         各ブロックのindexごとに冗長ヘッダを作成する
         実際のエンコード(と冗長データ作成)はencode_rs()
      */
 public:
  byte *encode(int len, byte d[]) {
    int turn_len = len / K;
    if (len % K != 0) turn_len++;
    byte *enc = new byte[N * turn_len];
    len_enc = N * turn_len;

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
};

int main(int argc, char *args[]) {
  broadcastImg(args[1]);

  int srcsocket;
  srcsocket = openTcp();
  if (srcsocket < 0) {
    return -1;
  }

  int maxfd = srcsocket;
  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(srcsocket, &rfds);

  for (int i = 0; i < MAX_DEVICE; i++) {
    member[i].fd = 0;
    member[i].addrSize = sizeof(member[i].addr);
    FD_SET(member[i].fd, &rfds);
    if (member[i].fd > maxfd) maxfd = member[i].fd;
  }

  std::thread threads[MAX_DEVICE];

  struct timeval timeout;
  timeout.tv_sec = 10;
  timeout.tv_usec = 0;
  int timeout_ = 1;

  printf("[2]Start receive images...\n");

  for (int i = 0; i < MAX_DEVICE; i++) {
    timeout_ = select(maxfd + 1, &rfds, NULL, NULL, &timeout);

    if (timeout_ == 0) {
      printf("timeout\n");
      break;
    }

    if (FD_ISSET(srcsocket, &rfds) &&
        (member[i].fd = accept(srcsocket, (struct sockaddr *)&member[i].addr,
                               &member[i].addrSize)) < 0) {
      perror("accept");
      close(srcsocket);
      return -1;
    }

    member[i].no = i;
    imgs[i] = cv::Mat(ROWS, COLS, CV_8UC3);
    threads[i] = std::thread(receiveImg, member[i]);  //ポインタ渡ししたかった
  }

  printf("-------------------------\n");
  printf("[3]Start feedback...\n");

  char key = 0;
  for (int i = 0; i < MAX_DEVICE; i++) {
    if (member[i].fd == 0) break;
    printf("%s\n", member[i].name);

    cv::imshow(member[i].name, imgs[i]);

    key = cv::waitKey(0);

    if (key == '1') member[i].feedback = 1;
    feedback(member[i]);
    cv::destroyWindow(member[i].name);
    threads[i].join();
  }

  close(srcsocket);

  return 0;
}

static int broadcastImg(char file_name[]) {
  static char buff[DATA_SIZE + HEAD_SIZE];  //送信するバッファ(ヘッダ部 +
                                            //分割された画像データ)

  int send_sock;
  struct sockaddr_in send_addr;
  int opt = 1;

  send_sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (send_sock < 0) {
    perror("socket");
    return -1;
  }

  send_addr.sin_family = AF_INET;
  send_addr.sin_port = htons(PORT);  // 送信ポート番号
  send_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);

  setsockopt(send_sock, SOL_SOCKET, SO_BROADCAST, (char *)&opt,
             sizeof(opt));  //ブロードキャストをするのに必要な設定

  // char file_name[100];
  static int divImageNum;   //画像を送信する際の分割数
  static int lastdataSize;  //分割された画像の最後のサイズ

  // printf("input file_name: ");
  // scanf("%s", file_name);
  // fflush(stdin);

  char threshold;
  int tmp = 80;

  // printf("input threshold: ");
  // scanf("%d", &tmp);
  threshold = (char)tmp;

  cv::Mat img = cv::imread(file_name, cv::IMREAD_COLOR);
  cv::resize(img, img, cv::Size(), COLS / img.cols,
             ROWS / img.rows);  //横、縦の順であることに注意

  divImageNum = ((img.rows * img.step) / DATA_SIZE) + 1;  //画像の分割数

  if (divImageNum > 127 * (HEAD_SIZE - 1)) {
    printf("Can't divide. Because the image is too large.\n");
    return -1;
  }
  printf("divImageNum=%d\n", divImageNum);
  lastdataSize = (img.rows * img.step) % DATA_SIZE;

  //誤り訂正符号生成の前準備
  for (int i = 0; i < divImageNum - 1; i++) {
    memcpy(&imgcode[i], &img.data[DATA_SIZE * i], DATA_SIZE);
  }
  memcpy(&imgcode[divImageNum - 1], &img.data[DATA_SIZE * (divImageNum - 1)],
         lastdataSize);

  ReedSolomon *rs = new ReedSolomon(8, 19);  // M=8 T=8 rs(255,239)

  for (int i = 0; i < DATA_SIZE; i++) {
    for (int j = 0; j < 255; j++) {
      tcode[i][j] = imgcode[j][i];
    }
  }

  for (int i = 0; i < DATA_SIZE; i++) {
    byte *enc = rs->encode(divImageNum, tcode[i]);

    for (int j = 0; j < len_enc; j++) {
      sendcode[j][i] = enc[j];
    }

    delete enc;
  }

  delete rs;

  printf("[1]Start Sending Image...\n");

  buff[0] = 0;
  buff[1] = threshold;
  setSeq(divImageNum, buff);
  sendto(send_sock, buff, DATA_SIZE + HEAD_SIZE, 0,
         (struct sockaddr *)&send_addr, sizeof(send_addr));
  memset(buff, 0, sizeof(buff));

  for (int i = 0; i < 255; i++) {
    buff[0] = 1;
    buff[1] = threshold;
    setSeq(i, buff);
    memcpy(&buff[HEAD_SIZE], &sendcode[i], DATA_SIZE);
    sendto(send_sock, buff, DATA_SIZE + HEAD_SIZE, 0,
           (struct sockaddr *)&send_addr, sizeof(send_addr));
    // usleep(SEND_TIME / 255);
  }

  close(send_sock);

  printf("Sending Ended.\n-------------------------\n");

  return 1;
}

static int openTcp() {
  int fd;
  int opt = 1;
  struct sockaddr_in addr;

  fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    perror("socket");
    return -1;
  }

  memset((char *)&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(PORT);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);

  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    perror("bind");
    close(fd);
    return -1;
  }

  if (listen(fd, MAX_DEVICE) < 0) {
    perror("listen");
    close(fd);
    return -1;
  }

  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt));

  return fd;
}

static void setSeq(int num, char *buff) {  // charは-128~127までしか取れない

  for (int i = 0; i < num / 127; i++) {
    buff[2 + i] = 127;
  }
  buff[2 + num / 127] = num % 127;

  return;
}

static int checkSeq(char *buff) {  // charは-128~127までしか取れない

  int num = 0;

  for (int i = 2; i < HEAD_SIZE; i++) {
    num += (int)buff[i];
  }

  return num;
}

void receiveImg(Mem mem) {
  int re = read(mem.fd, mem.buff,
                HEAD_SIZE + DATA_SIZE);  //デバイス名　分割数を呼び出す
  int divNum = checkSeq(mem.buff);

  memcpy(member[mem.no].name, &mem.buff[HEAD_SIZE], 100);
  memset(mem.buff, 0, sizeof(mem.buff));

  int tmpNum = 0;
  while (true) {
    re = read(mem.fd, mem.buff, DATA_SIZE);
    memcpy(&imgs[mem.no].data[tmpNum], &mem.buff, re);
    tmpNum += re;
    if ((tmpNum / DATA_SIZE) + 1 >= divNum) break;
  }

  return;
}

void feedback(Mem mem) {
  static char buff[DATA_SIZE + HEAD_SIZE];

  printf("feedback %s : %d\n", (mem.name), mem.feedback);

  buff[0] = 2;
  buff[1] = mem.feedback;
  write(mem.fd, buff, DATA_SIZE);
  close(mem.fd);
  return;
}

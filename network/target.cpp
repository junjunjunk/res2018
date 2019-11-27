#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <sys/ioctl.h>
#include <thread>
#include <vector>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <math.h>

#define ARRAY_LENGTH(array) (sizeof(array) / sizeof(array[0]))

static const float ROWS = 224;
static const float COLS = 224;
static char device_name[50];

static const int DATA_SIZE = 700;
static const int HEAD_SIZE = 10;
static const int PORT = 80;
static char buff[DATA_SIZE + HEAD_SIZE];

typedef unsigned char byte;

//排他制御や終了処理のフラグ
static bool end_flag_ = false;
static bool wait_set_flag_ = false;

//共有変数
static int re;
static int threshold;
static int len_ans;
static unsigned char imgcode[216][DATA_SIZE];
static unsigned char tcode[DATA_SIZE][255];
static unsigned char receivedcode[255][DATA_SIZE];

struct sockaddr_in senderinfo;
cv::Mat receiveimage(ROWS, COLS, CV_8UC3);

void ThreadReceive();
void ThreadStore();
int sendImage2Machine();
int accessMachine();
void correctImage();

class ReedSolomon
{
        int M = 8;   //GF(2^M) 1シンボル 8ビット
        int N = 255; //2^M -1 各ブロックの総シンボル数
        int T = 8;   // N-K = 2T　訂正能力　2Tは各ブロックに加えるパリティ・シンボル数
        int K = 239; // data　各ブロックの情報シンボル数

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
        ReedSolomon(int m, int n, int t, int k)
        {
                if ((m == 8 || m == 4))
                {
                        M = m;
                        T = t;
                        N = (int)(pow(2, M)) - 1;
                        K = N - (2 * T);
                        init();
                }
                else
                {
                        perror("M != (4 or 8)");
                        exit(0);
                }
        }

        /**
       m: GF(2^m)のM指定
       t: 誤り訂正能力Tの指定　のコンストラクタ
     */
      public:
        ReedSolomon(int m, int t)
        {
                if ((m == 8 || m == 4))
                {
                        M = m;
                        T = t;
                        N = (int)(pow(2, M)) - 1;
                        K = N - (2 * T);
                        init();
                }
                else
                {
                        perror("M != (4 or 8)");
                        exit(0);
                }
        }

        /**
       N=255固定 誤り訂正能力Tを設定するコンストラクタ
    */
      public:
        ReedSolomon(int t)
        {
                T = t;
                K = N - (2 * T);
                init();
        }

        /**
       N=255固定　誤り訂正能力T=8 固定のデフォルトコンストラクタ
    */
      public:
        ReedSolomon()
        {
                init();
        }

        void init()
        {
                G = new int[N - K + 1];
                alpha_to = new int[N + 1];
                index_of = new int[N + 1];
                int *P = (M == 8) ? P8 : P4;
                int i, j, mask;
                mask = 1;

                alpha_to[M] = 0;
                for (i = 0; i < M; i++)
                {
                        alpha_to[i] = mask;
                        index_of[alpha_to[i]] = i;
                        if (P[i] != 0)
                                alpha_to[M] ^= mask;
                        mask <<= 1;
                }
                index_of[alpha_to[M]] = M;
                mask >>= 1;
                for (i = M + 1; i < N; i++)
                {
                        if (alpha_to[i - 1] >= mask)
                                alpha_to[i] = alpha_to[M] ^ ((alpha_to[i - 1] ^ mask) << 1);
                        else
                                alpha_to[i] = alpha_to[i - 1] << 1;
                        index_of[alpha_to[i]] = i;
                }
                index_of[0] = -1;

                G[0] = 2;
                G[1] = 1;
                for (i = 2; i <= N - K; i++)
                {
                        G[i] = 1;
                        for (j = i - 1; j > 0; j--)
                                if (G[j] != 0)
                                        G[j] = G[j - 1] ^ alpha_to[(index_of[G[j]] + i) % N];
                                else
                                        G[j] = G[j - 1];
                        G[0] = alpha_to[(index_of[G[0]] + i) % N];
                }
                for (i = 0; i <= N - K; i++)
                        G[i] = index_of[G[i]];
                printf("M=%d\nN=%d\nT=%d\nK=%d\n", M, N, T, K);
        }

      public:
        int getM()
        {
                return M;
        }

      public:
        int getN()
        {
                return N;
        }

      public:
        int getK()
        {
                return K;
        }

      public:
        int getT()
        {
                return T;
        }

        /**
       デコードしてデータを並び替えて取り出す
       実際のデコードはdecode_rs()
    */
      public:
        byte *decode(int len, byte d[])
        {
                int turn_len = len / N;
                if (len % N != 0)
                        turn_len++; //たぶん実行されないはず
                byte *ans = new byte[K * turn_len];
                len_ans = K * turn_len;
                int *recd = new int[N];

                for (int i = 0; i < turn_len; i++)
                {
                        for (int j = 0; j < N - K; j++)
                                recd[j] = index_of[d[i + turn_len * j] & 0xFF];
                        for (int j = 0; j < K; j++)
                                recd[j + N - K] = index_of[d[i + turn_len * (N - K + j)] & 0xFF];
                        recd = decode_rs(N, recd);
                        for (int j = 0; j < K; j++)
                                ans[i + turn_len * j] = (byte)recd[j + N - K];
                }
                return ans;
        }

        /**
       デコードを行う Nシンボル→Nシンボル
       冗長データの除去は別途行う必要がある
    */
      public:
        int *decode_rs(int len, int recd[])
        {
                int i, j, u, q;
                // int *elp = new int[N-K+2][N-K];
                std::vector<std::vector<int>> elp;
                elp.resize(N - K + 2);
                for (int i = 0; i < N - K + 2; i++)
                {
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

                if (len != N)
                {
                        perror("decode size error");
                        return nullptr;
                }
                /* first form the syndromes */
                for (i = 1; i <= N - K; i++)
                {
                        s[i] = 0;
                        for (j = 0; j < N; j++)
                                if (recd[j] != -1)
                                        s[i] ^= alpha_to[(recd[j] + i * j) % N];
                        if (s[i] != 0)
                                syn_error = 1;
                        s[i] = index_of[s[i]];
                }

                if (syn_error > 0)
                { /* エラーがあった場合訂正を試みる */
                        d[0] = 0;
                        d[1] = s[1];
                        elp[0][0] = 0;
                        elp[1][0] = 1;
                        for (i = 1; i < N - K; i++)
                        {
                                elp[0][i] = -1;
                                elp[1][i] = 0;
                        }
                        l[0] = 0;
                        l[1] = 0;
                        u_lu[0] = -1;
                        u_lu[1] = 0;
                        u = 0;

                        do
                        {
                                u++;
                                if (d[u] == -1)
                                {
                                        l[u + 1] = l[u];
                                        for (i = 0; i <= l[u]; i++)
                                        {
                                                elp[u + 1][i] = elp[u][i];
                                                elp[u][i] = index_of[elp[u][i]];
                                        }
                                }
                                else
                                {
                                        q = u - 1;
                                        while ((d[q] == -1) && (q > 0))
                                                q--;
                                        if (q > 0)
                                        {
                                                j = q;
                                                do
                                                {
                                                        j--;
                                                        if ((d[j] != -1) && (u_lu[q] < u_lu[j]))
                                                                q = j;
                                                } while (j > 0);
                                        }
                                        if (l[u] > l[q] + u - q)
                                                l[u + 1] = l[u];
                                        else
                                                l[u + 1] = l[q] + u - q;

                                        for (i = 0; i < N - K; i++)
                                                elp[u + 1][i] = 0;
                                        for (i = 0; i <= l[q]; i++)
                                                if (elp[q][i] != -1)
                                                        elp[u + 1][i + u - q] = alpha_to[(d[u] + N - d[q] + elp[q][i]) % N];
                                        for (i = 0; i <= l[u]; i++)
                                        {
                                                elp[u + 1][i] ^= elp[u][i];
                                                elp[u][i] = index_of[elp[u][i]];
                                        }
                                }
                                u_lu[u + 1] = u - l[u + 1];

                                if (u < N - K)
                                {
                                        if (s[u + 1] != -1)
                                                d[u + 1] = alpha_to[s[u + 1]];
                                        else
                                                d[u + 1] = 0;
                                        for (i = 1; i <= l[u + 1]; i++)
                                                if ((s[u + 1 - i] != -1) && (elp[u + 1][i] != 0))
                                                        d[u + 1] ^= alpha_to[(s[u + 1 - i] + index_of[elp[u + 1][i]]) % N];
                                        d[u + 1] = index_of[d[u + 1]];
                                }
                        } while ((u < N - K) && (l[u + 1] <= T));

                        u++;
                        if (l[u] <= T)
                        {
                                for (i = 0; i <= l[u]; i++)
                                        elp[u][i] = index_of[elp[u][i]];

                                for (i = 1; i <= l[u]; i++)
                                        reg[i] = elp[u][i];
                                count = 0;
                                for (i = 1; i <= N; i++)
                                {
                                        q = 1;
                                        for (j = 1; j <= l[u]; j++)
                                                if (reg[j] != -1)
                                                {
                                                        reg[j] = (reg[j] + j) % N;
                                                        q ^= alpha_to[reg[j]];
                                                }
                                        if (!(q > 0))
                                        {
                                                root[count] = i;
                                                loc[count] = N - i;
                                                count++;
                                        }
                                }
                                if (count == l[u])
                                {
                                        for (i = 1; i <= l[u]; i++)
                                        {
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

                                        for (i = 0; i < N; i++)
                                        {
                                                err[i] = 0;
                                                if (recd[i] != -1)
                                                        recd[i] = alpha_to[recd[i]];
                                                else
                                                        recd[i] = 0;
                                        }
                                        for (i = 0; i < l[u]; i++)
                                        {
                                                err[loc[i]] = 1;
                                                for (j = 1; j <= l[u]; j++)
                                                        if (z[j] != -1)
                                                                err[loc[i]] ^= alpha_to[(z[j] + j * root[i]) % N];
                                                if (err[loc[i]] != 0)
                                                {
                                                        err[loc[i]] = index_of[err[loc[i]]];
                                                        q = 0;
                                                        for (j = 0; j < l[u]; j++)
                                                                if (j != i)
                                                                        q += index_of[1 ^ alpha_to[(loc[j] + root[i]) % N]];
                                                        q = q % N;
                                                        err[loc[i]] = alpha_to[(err[loc[i]] - q + N) % N];
                                                        recd[loc[i]] ^= err[loc[i]];
                                                }
                                        }
                                }
                                else
                                        for (i = 0; i < N; i++)
                                                if (recd[i] != -1)
                                                        recd[i] = alpha_to[recd[i]];
                                                else
                                                        recd[i] = 0;
                        }
                        else
                                for (i = 0; i < N; i++)
                                        if (recd[i] != -1)
                                                recd[i] = alpha_to[recd[i]];
                                        else
                                                recd[i] = 0;
                }
                else
                        for (i = 0; i < N; i++)
                                if (recd[i] != -1)
                                        recd[i] = alpha_to[recd[i]];
                                else
                                        recd[i] = 0;
                return recd;
        }
};

int main(int argc, char *args[])
{

        printf("'iphone'\n'keisandoor'\n'keisanmac'\n'keisanmonitor'\n'kisoair'\n'kisocoffee'\n'kisocoffee'\n'kisodoor'\n'kisolight'\n'kisoprinter'\n'macbook'\n");
        printf("select:");
        scanf("%s", device_name);

        std::thread th_recv(ThreadReceive);
        std::thread th_stor(ThreadStore);

        th_recv.join();
        th_stor.join();

        ReedSolomon *rs = new ReedSolomon(8, 19); //M=8 T=8 rs(255,239)

        for (int i = 0; i < DATA_SIZE; i++)
        {
                for (int j = 0; j < 255; j++)
                {
                        tcode[i][j] = receivedcode[j][i];
                }
        }

        for (int i = 0; i < DATA_SIZE; i++)
        {
                byte *ans = rs->decode(255, tcode[i]);
                for (int j = 0; j < 216; j++)
                {
                        imgcode[j][i] = ans[j];
                }
                delete ans;
        }

        for (int i = 0; i < 216; i++)
        {
                memcpy(&receiveimage.data[DATA_SIZE * i], imgcode[i], DATA_SIZE);
        }

        cv ::imwrite("./shell/received.bmp", receiveimage);

        delete rs;
        // if(sendImage2Machine()==0){
        //         printf("Cannot send Image to DL machine.\n");
        //         return 0;
        // }

        // int result = accessMachine();

        // printf("predict:%d,threshold:%d\n-------------------------\n", result, threshold);

        // if (result < threshold)
        // {
        //         printf("Do not respond to user.\n");
        //         return 0;
        // }
        // else
        // {
        //         printf("Respond to user.\n");
        // }

        printf("[2]Start responsing to user.\n");

        int resp_sock;
        resp_sock = socket(AF_INET, SOCK_STREAM, 0);
        senderinfo.sin_port = htons(PORT);

        printf("connecting... IP=%s　port=%d\n", inet_ntoa(senderinfo.sin_addr), ntohs(senderinfo.sin_port));
        connect(resp_sock, (struct sockaddr *)&senderinfo, sizeof(senderinfo));

        cv::Mat myimg = cv::imread("myimg.JPG", cv::IMREAD_COLOR);
        cv::resize(myimg, myimg, cv::Size(), COLS / myimg.cols, ROWS / myimg.rows); //横、縦の順であることに注意

        int divImageNum = ((myimg.rows * myimg.step) / DATA_SIZE) + 1;

        if (divImageNum > 127 * (HEAD_SIZE - 1))
        {
                printf("Can't divide. Because the image is too large.\n");
                return -1;
        }

        buff[0] = 0;
        for (int j = 0; j < divImageNum / 127; j++)
        {
                buff[2 + j] = 127; //分割された何番目か
        }

        buff[2 + divImageNum / 127] = divImageNum % 127;

        memcpy(&buff[HEAD_SIZE], device_name, DATA_SIZE);
        send(resp_sock, buff, HEAD_SIZE + DATA_SIZE, 0);
        memset(buff, 0, sizeof(buff));

        for (int i = 0; i < divImageNum - 1; i++)
        {
                memcpy(&buff, &myimg.data[DATA_SIZE * i], DATA_SIZE);
                send(resp_sock, buff, DATA_SIZE, 0);
        }

        memcpy(&buff, &myimg.data[DATA_SIZE * (divImageNum - 1)], (myimg.rows * myimg.step) % DATA_SIZE);
        send(resp_sock, buff, (myimg.rows * myimg.step) % DATA_SIZE, 0);

        printf("End sending message.\n");

        printf("-------------------------\n");

        printf("[3]Waiting feedback from user...\n");
        recv(resp_sock, buff, HEAD_SIZE + DATA_SIZE, 0);
        printf("result = %d\n", (int)buff[1]);
        printf("-------------------------\n");

        close(resp_sock);

        return 0;
}

void ThreadReceive()
{
        static int receive_sock;
        struct sockaddr_in receive_addr;
        static const int val = 1;
        socklen_t addrlen = sizeof(senderinfo);

        receive_sock = socket(AF_INET, SOCK_DGRAM, 0);

        receive_addr.sin_family = AF_INET;
        receive_addr.sin_port = htons(PORT);
        receive_addr.sin_addr.s_addr = INADDR_ANY;

        bind(receive_sock, (struct sockaddr *)&receive_addr, sizeof(receive_addr));
        ioctl(receive_sock, 0x5421, &val);

        fd_set readfds, fds;
        FD_ZERO(&readfds);
        FD_SET(receive_sock, &readfds);

        struct timeval timeout;
        timeout.tv_sec = 7;
        timeout.tv_usec = 0;
        int timeout_ = 1;

        printf("[1]Waiting to receive...\n");

        while (!end_flag_)
        {
                // memcpy(&fds, &readfds, sizeof(fd_set));

                timeout_ = select(receive_sock + 1, &readfds, NULL, NULL, &timeout);
                if (timeout_ == 0)
                {
                        // printf("timeout\n");
                        end_flag_ = true;
                        break;
                }

                if (wait_set_flag_)
                        continue; //バッファが格納待ちの場合は受信しない

                re = recvfrom(receive_sock, buff, DATA_SIZE + HEAD_SIZE, 0, (struct sockaddr *)&senderinfo, &addrlen);
                if (re > 0)
                        wait_set_flag_ = true;
        }

        printf("Receiving Ended.\n");
}

void ThreadStore()
{
        static int divNum, recNum, tmpNum; //div:分割総数　recNum:受信待ちの分割番号

        recNum = 0;

        while (!end_flag_)
        {

                if (!wait_set_flag_)
                        continue; //バッファが格納待ちではない場合処理を飛ばす

                switch ((int)buff[0])
                {

                case 0: //いくつ画像を分割して送るかの情報を受信した場合
                        divNum = -1;
                        for (int i = 1; i < HEAD_SIZE; i++)
                        {
                                divNum += (int)buff[i];
                        }
                        if (divNum > 0)
                        {
                                printf("User(Sender)info : %s\n", inet_ntoa(senderinfo.sin_addr));
                        }
                        wait_set_flag_ = false;
                        break;

                case 1: //分割された画像が送られてきた場合
                        tmpNum = 0;
                        for (int i = 2; i < HEAD_SIZE; i++)
                        {
                                tmpNum += (int)buff[i];
                        }
                        // printf("tmpNum=%d\n", tmpNum);
                        if (threshold == 0)
                        {
                                threshold = buff[1];
                                printf("threshold:%d\n", threshold);
                        }

                        if (tmpNum >= recNum)
                        {
                                memcpy(&receivedcode[tmpNum][0], &buff[HEAD_SIZE], re - HEAD_SIZE);
                                wait_set_flag_ = false;
                                recNum += 1;
                        }
                        break;

                default:
                        break;
                }
        }
}

int sendImage2Machine()
{
        FILE *fp;
        static char command[100] = "sh ./shell/scp_";
        const char dsh[] = ".sh";

        strcat(command, device_name);
        strcat(command, dsh);

        if ((fp = popen(command, "r")) == NULL)
        {
                fprintf(stderr, "パイプのオープンに失敗しました！");
                return 0;
        }
        pclose(fp);

        return 1;
}

int accessMachine()
{
        FILE *fp;
        static char command[100] = "sh ./shell/";
        const char dsh[] = ".sh";

        strcat(command, device_name);
        strcat(command, dsh);
        char buf[256];
        std::string tmp;

        int result;
        float percent;

        if ((fp = popen(command, "r")) == NULL)
        {
                fprintf(stderr, "パイプのオープンに失敗しました！");
                return 0;
        }

        while (fgets(buf, sizeof(buf), fp) != NULL)
        {
                if (strstr(buf, "percent") != NULL)
                {
                        tmp = strchr(buf, (int)':') + sizeof(char);
                        printf("%s", tmp.c_str());
                        break;
                }
        }

        pclose(fp);

        return (int)(stof(tmp));
}

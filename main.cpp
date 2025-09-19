// main.cpp
/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <sstream> 
#include "Timer.h"
#include "Vanity.h"
#include "SECP256k1.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include <csignal>
// 8891689_FIX: 引入 CUDA runtime 頭文件以檢查 GPU 設備
#include <cuda_runtime.h> 

#if defined(_WIN32) || defined(_WIN64)
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#endif

// ------------------- 鍵盤監聽和全局變量  -------------------
std::atomic<bool> Pause(false);
std::atomic<bool> Paused(false);
std::atomic<bool> stopMonitorKey(false);
int idxcount = 0;
double t_Paused = 0.0;
bool randomMode = false;
bool backupMode = false;

using namespace std;

VanitySearch* g_vanity_search_ptr = nullptr;
std::atomic<bool> g_shutdown_initiated(false);

void signalHandler(int signum) {
    if (!backupMode) {
        printf("\n"); 
        fflush(stdout); 
        exit(signum);
    }

    if (g_shutdown_initiated.exchange(true)) {
        exit(signum);
    }
    
    cout << "\n[!] Ctrl+C Detected. Shutting down gracefully, please wait...";
    cout.flush();
    
    if (g_vanity_search_ptr != nullptr) {
        g_vanity_search_ptr->endOfSearch = true;
    }
}

#if defined(_WIN32) || defined(_WIN64)
void monitorKeypress() {
	while (!stopMonitorKey) {
		Timer::SleepMillis(1);
		if (_kbhit()) {
			char ch = _getch();
			if (ch == 'p' || ch == 'P') {
				Pause = !Pause;
			}
		}
	}
}
#else
struct termios original_termios;
bool terminal_mode_changed = false;
void restoreTerminalMode() {
    if (terminal_mode_changed) {
        tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
    }
}
void setupRawTerminalMode() {
    tcgetattr(STDIN_FILENO, &original_termios);
    terminal_mode_changed = true;
    struct termios new_termios = original_termios;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}
void monitorKeypress() {
	while (!stopMonitorKey) {
		Timer::SleepMillis(1);
		char ch;
		if (read(STDIN_FILENO, &ch, 1) > 0) {
			if (ch == 'p' || ch == 'P') {
				Pause = !Pause;
			}
		}
	}
}
#endif

// ------------------- 輔助函數 -------------------

void printHelp() {
    printf("Usage: ./kk -r <bits> [-a <b58_addr> | -p <pubkey>] [options]\n\n");
    
    printf("Modes (choose one):\n");
    printf("  -a <b58_addr>       Find the private key for a P2PKH Bitcoin address.\n");
    printf("  -p <pubkey>         Find the private key for a specific public key (hex, compressed format only).\n\n");
    
    printf("Keyspace:\n");
    printf("  -r <bits>           Set the bit range for the search (e.g., 71 for 2^70 to 2^71-1) (required).\n\n");
    
    printf("Options:\n");
    printf("  -R                  Activate random mode.\n");
    printf("  -b                  Enable backup mode to resume from last progress (not for random mode).\n");
    printf("  -G <ID>             Specify the GPU ID to use, default is 0.\n");
    printf("  -h, --help          Display this help message.\n\n");
    
    printf("Technical Support: github.com/8891689\n");
    exit(0);
}

int getInt(string name, char* v) {
	int r;
	try { r = std::stoi(string(v)); }
	catch (std::invalid_argument&) {
		fprintf(stderr, "[ERROR] Invalid %s argument, number expected\n", name.c_str());
		exit(-1);
	}
	return r;
}

bool loadBackup(int& idxcount, double& t_Paused, int gpuid) {
    string filename = "schedule_gpu" + to_string(gpuid) + ".dat";
    ifstream inFile(filename, std::ios::binary);
    if (inFile) {
        inFile.read(reinterpret_cast<char*>(&idxcount), sizeof(idxcount));
        inFile.read(reinterpret_cast<char*>(&t_Paused), sizeof(t_Paused));
        inFile.close();
        return true;
    }
    return false;
}

// ------------------- Main 函數 (核心修改區) -------------------

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);

#if !(defined(_WIN32) || defined(_WIN64))
    atexit(restoreTerminalMode);
    setupRawTerminalMode();
#endif

    std::thread inputThread(monitorKeypress);
    Timer::Init();
    Secp256K1* secp = new Secp256K1();
    secp->Init();

    if (argc < 2) {
        printHelp();
    }
    
    string target_address;
    string target_pubkey;
    int bits = 0;
    int gpuId = 0; // 默認使用 GPU 0
    uint32_t maxFound = 65536 * 4;

    // 參數解析循環
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp();
        } else if (arg == "-b") {
            backupMode = true;
        } else if (arg == "-R") {
            randomMode = true;
        } else if (arg == "-a") {
            if (i + 1 < argc) { target_address = argv[++i]; } 
            else { fprintf(stderr, "[ERROR] An address value is required after the -a parameter.\n"); exit(-1); }
        } else if (arg == "-p") {
            if (i + 1 < argc) { target_pubkey = argv[++i]; }
            else { fprintf(stderr, "[ERROR] A public key hex string is required after the -p parameter.\n"); exit(-1); }
        } else if (arg == "-r") {
            if (i + 1 < argc) {
                bits = getInt((char*)"-r", argv[++i]);
                if (bits <= 0 || bits > 256) { fprintf(stderr, "[ERROR] -r value (number of bits) must be between 1 and 256.\n"); exit(-1); }
            } else { fprintf(stderr, "[ERROR] A numeric value is required after the -r parameter.\n"); exit(-1); }
        } else if (arg == "-G") {
            if (i + 1 < argc) gpuId = getInt((char*)"-G", argv[++i]);
        } else {
            fprintf(stderr, "[ERROR] Unknown parameter: %s\n", arg.c_str());
            printHelp();
        }
    }
    
    // 參數驗證邏輯
    if ((target_address.empty() && target_pubkey.empty()) || bits == 0) {
        fprintf(stderr, "[ERROR] Either address (-a) or public key (-p), and range (-r) must be specified.\n");
        printHelp();
    }
    if (!target_address.empty() && !target_pubkey.empty()) {
        fprintf(stderr, "[ERROR] Cannot use -a and -p at the same time. Please choose one.\n");
        printHelp();
    }
    if (backupMode && randomMode) {
        fprintf(stderr, "[ERROR] Backup mode (-b) cannot be used with random mode (-R).\n");
        exit(-1);
    }
    
    // =========================================================================
    // 8891689_FIX: 在此處添加 GPU 設備驗證，這是解決問題的關鍵
    // =========================================================================
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA error while checking for devices: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "[INFO] Please ensure NVIDIA drivers and CUDA toolkit are installed correctly.\n");
        exit(-1);
    }

    if (deviceCount == 0) {
        // 修改爲打印信息沒檢測到，並退出
        fprintf(stdout, "[INFO] No CUDA-enabled GPU was detected. Exiting.\n");
        exit(0);
    }

    if (gpuId >= deviceCount || gpuId < 0) {
        // 修改爲打印信息沒檢測到，並退出
        fprintf(stdout, "[INFO] Invalid GPU ID %d specified. No device detected with this ID.\n", gpuId);
        fprintf(stdout, "[INFO] Detected %d GPU(s). Valid IDs are from 0 to %d.\n", deviceCount, deviceCount - 1);
        exit(0);
    }
    // =========================================================================
    // GPU 驗證結束
    // =========================================================================


    vector<string> target_vector;
    string search_target_display; 
    if (!target_pubkey.empty()) {
        target_vector.push_back(target_pubkey);
        search_target_display = target_pubkey;
    } else {
        target_vector.push_back(target_address);
        search_target_display = target_address;
    }

    BITCRACK_PARAM bitcrack, *bc;
    bc = &bitcrack;
    bc->ksStart.SetInt32(1);
    if (bits > 1) {
        bc->ksStart.ShiftL(bits - 1);
    }
    bc->ksFinish.SetInt32(1);
    bc->ksFinish.ShiftL(bits);
    bc->ksFinish.SubOne();
    bc->ksNext.Set(&bc->ksStart);

    if (backupMode) {
        if (loadBackup(idxcount, t_Paused, gpuId)) {
            printf("[+] Restoring from backup was successful. Starting batch: %d, Elapsed time: %.2f s.\n", idxcount, t_Paused);
        } else {
            printf("[+] Backup file not found. Will start from scratch.\n");
        }
    }
    
    printf("[+] KeyKiller v.007\n");
    if (!target_pubkey.empty()) {
        printf("[+] Search: %s [Public Key]\n", search_target_display.c_str());
    } else {
        printf("[+] Search: %s [P2PKH/Compressed]\n", search_target_display.c_str());
    }
    time_t now = time(NULL);
    printf("[+] Start %s", ctime(&now));
    if (randomMode) printf("[+] Random mode\n");
    printf("[+] Range (2^%d)\n", bits);
    printf("[+] from : 0x%s\n", bc->ksStart.GetBase16().c_str());
    printf("[+] to   : 0x%s\n", bc->ksFinish.GetBase16().c_str());
    fflush(stdout);

    VanitySearch* v = new VanitySearch(secp, target_vector, SEARCH_COMPRESSED, true, "", maxFound, bc);
    g_vanity_search_ptr = v; 
    //改進自動調節網格的大小。
    vector<int> gpuIds = { gpuId };
    vector<int> gridSizes = { -1, 128 }; 
    
    v->Search(gpuIds, gridSizes);

    stopMonitorKey = true;
    if (inputThread.joinable()) {
        inputThread.join();
    }
    printf("\n");
    delete v;
    delete secp;
    return 0;
}

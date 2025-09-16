# keyKiller Cuda

keyKiller is the GPU-powered version of the keyKiller project, designed to achieve extreme performance in solving Satoshi puzzles on modern NVIDIA GPUs. 
Leveraging CUDA, warp-level parallelism, and batch EC operations, keyKiller CUDA pushes the limits of cryptographic key search.

1. The Secp256k1 algorithm is based on the excellent work of [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch) ， [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul)，[KeyHunt-Cuda](https://github.com/Qalander/KeyHunt-Cuda), [CUDACyclone](https://github.com/Dookoo2/CUDACyclone) ，This implementation is inspired and referenced by the above implementations. Contributions are welcome! The algorithm has been significantly modified for CUDA. Special thanks to Jean-Luc Pons for his pioneering contributions to the cryptography community.

2. keyKiller GPU-based solution to Satoshi's puzzle. This is an experimental project, Please look at it rationally! 

3. While keyKiller CUDA is simple to use, it leverages massive GPU parallelism** to achieve extreme performance in elliptic curve calculations, compressed public keys, and Hash160 pipelines.

4. Theoretically, the best configuration for 4090 is -g 128,128 -s 16, but this needs to be tested on the actual platform. Each platform environment is different and the results obtained are also different. It is best to adjust it yourself and use the -g value that is fastest!


## Key Features

1. GPU Acceleration: Optimized for NVIDIA GPUs with full CUDA support.
2. Massive Parallelism: Tens of thousands of threads computing elliptic curve points and hash160 simultaneously.
3. Batch EC Operations: Efficient group addition and modular inversion with warp-level optimizations.
4. Grid/Batch Control: Fully configurable GPU execution with `-g` parameter (threads per batch × points per batch).
5. Cross-Platform: Works on Linux and Windows (via WSL2 or MinGW cross-compilation).
6. Cross Architecture: Automatic compilation for different architectures (75 86 89 90).
7. Extremely low VRAM usage: Key feature! For low price rented GPU.

## User Manual
```bash
./kk -h
Usage: ./kk -r <start_hex>:<end_hex> {-a <b58> | -h <hash160_hex> | -p <pubkey_hex>} [-R N] [-g A,B] [-s N] [-help|--help]

Modes (choose one target type):
  -a <b58_addr>              : Find private key for a P2PKH Bitcoin address.
  -h <hex>                   : Find private key for a 160-bit hash (RIPEMD160(SHA256(pubkey))).
  -p <hex>                   : Find private key for a 33-byte compressed public key.

Search Options:
  -r <start>:<end>           : Hex range of private keys to search (must be power of 2).
  -R <N>                     : Random search mode. N is the number of keys in millions
                               per random starting point. Runs indefinitely.

Performance Tuning:
  -g <A,B>                   : Set points batch size (A) and batches per SM (B).
                               A must be a power of two. Default: 128,8
  -s <N>                     : Set number of batches per kernel launch. Default: 64

Help:
-help , --help               : Show this help message！Technical support : github.com/8891689

```
## Options
- **-r**: range of search. Must be a power of two!
- **-a**: Given a P2PKH Bitcoin address.
- **-h**: Given a 160-bit hash (RIPEMD160(SHA256(pubkey).
- **-p**: Given a public key, find the private key of the 33-byte compressed public key。
- **-g**: very usefull parameter. Example -g 512,512 - first 512 - number of points each thread will process in one batch (Points batch size)., second 512 - number of threads in one group (Threads per batch).
- **-s**: batch per thread for one kernel launch.
- **-R**: Random search mode. N is the number of keys in millions per random starting point. Runs indefinitely.

## Example Output

Below is a sample run of keyKiller for reference.

**RTX4060**

```bash
./kk -r 2000000000:3FFFFFFFFF -a 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2 -g 512,256
GPU Information      : PASS 
Device               : NVIDIA GeForce RTX 4060 (compute 8.9)
SM                   : 24
ThreadsPerBlock      : 256
Blocks               : 4096
Points batch size    : 512
Batches/SM           : 256
Memory utilization   : 6.9% (538.3 MB / 7.63 GB) 
Total threads        : 1048576
Time: 8.0 s | Speed: 1268.9 Mkeys/s | Count: 10204470016 | Progress: 7.42 %

Gong Xi Fa Cai       ：Matching success
Key Hex              : 00000000000000000000000000000000000000000000000000000022382FACD0
Pub Hex              : 03C060E1E3771CBECCB38E119C2414702F3F5181A89652538851D2E3886BDD70C6
```

**RTX4090**
```bash
./kk -r 200000000000:3fffffffffff -a 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP -g 128,128 -s 16
GPU Information      : PASS 
Device               : NVIDIA GeForce RTX 4090 (compute 8.9)
SM                   : 128
ThreadsPerBlock      : 256
Blocks               : 16384
Points batch size    : 128
Batches/SM           : 128
Batches/launch       : 16 (per thread)
Memory utilization   : 4.8% (1.14 GB / 23.6 GB)
Total threads        : 4194304
mode 2               : Incremental mode
Time: 393.7 s | Speed: 6127.4 Mkeys/s | Count: 2421341587872 | Progress: 6.88 %

Gong Xi Fa Cai       ：Matching success
Key Hex              : 00000000000000000000000000000000000000000000000000002EC18388D544
Pub Hex              : 03FD5487722D2576CB6D7081426B66A3E2986C1CE8358D479063FB5F2BB6DD5849
```
**RTX5090**
```bash
./kk -r 200000000000:3fffffffffff -a 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP -g 128,256
GPU Information      : PASS 
Device               : NVIDIA GeForce RTX 5090 (compute 12.0)
SM                   : 170
ThreadsPerBlock      : 256
Blocks               : 1024
Points batch size    : 128
Batches/SM           : 8
Memory utilization   : 1.7% (557.3 MB / 31.4 GB) 
Total threads        : 262144
mode 2               : Incremental mode
Time: 7.0 s | Speed: 8408.0 Mkeys/s | Count: 58545467200 | Progress: 0.17 % ^C

```
**RTX3070 mobile**
```bash
./kk -r 2000000000:3FFFFFFFFF -a 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2 -g 512,256
GPU Information      : PASS 
Device               : NVIDIA GeForce RTX 3070 Laptop GPU (compute 8.6)
SM                   : 40
ThreadsPerBlock      : 256
Blocks               : 8192
Points batch size    : 512
Batches/SM           : 256
Batches/launch       : 64 (per thread)
Memory utilization   : 64.0% (5.12 GB / 8.00 GB)
Total threads        : 2097152
mode 2               : Incremental mode
Time: 61.2 s | Speed: 1234.3 Mkeys/s | Count: 72707573152 | Progress: 52.90 %

Gong Xi Fa Cai       ：Matching success
Pub Hex              : 00000000000000000000000000000000000000000000000000000022382FACD0
Pub Hex              : 03C060E1E3771CBECCB38E119C2414702F3F5181A89652538851D2E3886BDD70C6
```
## Compile

```bash
apt update;
apt-get install -y joe;
apt-get install -y zip;
apt-get install -y screen;
apt-get install -y curl libcurl4;
apt-get install build-essential;
apt-get install -y gcc;
apt-get install -y make;
apt install cuda-toolkit;
make
git clone https://github.com/8891689/keyKiller-Cuda.git
```
## Community benchmarks

1. Address and HASH160 Mode
```bash

| GPU               | Grid      | Speed (Mkeys/s) | Notes        |
| RTX1030           | 512,512   | 49.4 Mkeys/s    | My test      |

2. Public Key Mode

| GPU               | Grid      | Speed (Mkeys/s) | Notes        |
| RTX1030           | 512,512   | 99.6 Mkeys/s    | My test      |

```
# Sponsorship
If this project has been helpful or inspiring, please consider buying me a coffee. Your support is greatly appreciated. Thank you!
```
BTC: bc1qt3nh2e6gjsfkfacnkglt5uqghzvlrr6jahyj2k
ETH: 0xD6503e5994bF46052338a9286Bc43bC1c3811Fa1
DOGE: DTszb9cPALbG9ESNJMFJt4ECqWGRCgucky
TRX: TAHUmjyzg7B3Nndv264zWYUhQ9HUmX4Xu4
```

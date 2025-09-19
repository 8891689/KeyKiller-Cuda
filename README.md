# KeyKiller Cuda

KeyKiller is the GPU-powered version of the KeyKiller project, designed to achieve extreme performance in solving Satoshi puzzles on modern NVIDIA GPUs. 
Leveraging CUDA, warp-level parallelism, and batch EC operations, KeyKiller CUDA pushes the limits of cryptographic key search.

1. The Secp256k1 algorithm is based on the excellent work of [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch) ， [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul)，This implementation is based on modifications and improvements of the above implementation. Contributions are welcome! The algorithm has been significantly modified for CUDA. Special thanks to Jean-Luc Pons for his pioneering contributions to the cryptography community.

2. KeyKiller GPU-based solution to Satoshi's puzzle. This is an experimental project, Please look at it rationally! 

3. While KeyKiller CUDA is simple to use, it leverages massive GPU parallelism** to achieve extreme performance in elliptic curve calculations, compressed public keys, and Hash160 pipelines.

4. In theory, 4090 automatically configures the size, and the theoretical speed is about 6G, but this needs to be tested on the actual platform. Each platform environment is different, and the results obtained are also different. 
5. This program is still in the testing stage and may have unknown issues. It will continue to be improved and deeply optimized.

## Key Features

1. GPU Acceleration: Optimized for NVIDIA GPUs with full CUDA support.
2. Massive Parallelism: Tens of thousands of threads computing elliptic curve points and hash160 simultaneously.
3. Batch EC Operations: Efficient group addition and modular inversion with warp-level optimizations.
4. Grid/Batch Control: Use GPU execution with automatically configured parameters (number of threads per batch × number of points per batch).
5. Cross-Platform: Works on Linux and Windows .
6. -R command random mode does not slow down, high-speed calculation.
7. Incremental mode, with -b breakpoint save progress mode so you can continue working when you have time.


## User Manual
```bash
./kk -h
Usage: ./kk -r <bits> [-a <b58_addr> | -p <pubkey>] [options]

Modes (choose one):
  -a <b58_addr>       Find the private key for a P2PKH Bitcoin address.
  -p <pubkey>         Find the private key for a specific public key (hex, compressed format only).

Keyspace:
  -r <bits>           Set the bit range for the search (e.g., 71 for 2^70 to 2^71-1) (required).

Options:
  -R                  Activate random mode.
  -b                  Enable backup mode to resume from last progress (not for random mode).
  -G <ID>             Specify the GPU ID to use, default is 0.
  -h, --help          Display this help message.

Technical Support: github.com/8891689

```
## Options
- **-a**: Given a P2PKH Bitcoin address, crack its private key.
- **-p**: Given a public key, crack its private key. It must be a compressed public key.
- **-r**: range of search. Must be a power of two!Set the bit range for the search (e.g., 71 for 2^70 to 2^71-1) (required).
- **-R**: Activate random mode.
- **-b**: Enable backup mode to resume from last progress (not for random mode).
- **-G**: Specify the GPU ID to use, default is 0.
-  **p**: Press the p key to pause your work and press it again to resume it.

## Example Output

Below is a sample run of KeyKiller for reference.

**RTX1030**

```bash
./kk -r 33 -a 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu
[+] KeyKiller v.007
[+] Search: 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu [P2PKH/Compressed]
[+] Start Fri Sep 19 04:40:53 2025
[+] Range (2^33)
[+] from : 0x100000000
[+] to   : 0x1FFFFFFFF
[+] GPU: GPU #0 NVIDIA GeForce GT 1030 (3x128 cores) Grid(256x256)
[+] Starting keys set in 0.02 seconds
[+] GPU 57.61 Mkey/s][Total 2^31.39][Prob 65.62%] [50% in seconds][Found 0]  

[!] (Add): 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu
[!] (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9MDGKrXXQL647jj
[!] (HEX): 0x00000000000000000000000000000000000000000000000000000001A96CA8D8

```

**RTX1030**
```bash
./kk -r 33 -a 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu -R
[+] KeyKiller v.007
[+] Search: 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu [P2PKH/Compressed]
[+] Start Fri Sep 19 04:41:56 2025
[+] Random mode
[+] Range (2^33)
[+] from : 0x100000000
[+] to   : 0x1FFFFFFFF
[+] GPU: GPU #0 NVIDIA GeForce GT 1030 (3x128 cores) Grid(384x256)
[+] Starting keys set in 0.03 seconds
[+] [GPU 56.93 Mkey/s][Total 2^31.94][Prob 9.6e+01%][50% in seconds][Found 0]  

[!] (Add): 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu
[!] (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9MDGKrXXQL647jj
[!] (HEX): 0x00000000000000000000000000000000000000000000000000000001A96CA8D8

```
**RTX1030**
```bash
./kk -r 31 -p 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28
[+] KeyKiller v.007
[+] Search: 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28 [Public Key]
[+] Start Fri Sep 19 04:59:25 2025
[+] Range (2^31)
[+] from : 0x40000000
[+] to   : 0x7FFFFFFF
[+] GPU: GPU #0 NVIDIA GeForce GT 1030 (3x128 cores) Grid(256x256)
[+] Starting keys set in 0.02 seconds
[+] GPU 100.04 Mkey/s][Total 2^29.17][Prob 56.25%] [50% in seconds][Found 0]  

[!] (Pub): 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28
[!] (WIF): Compressed:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M9SmFMSCA4jQRW
[!] (HEX): 0x000000000000000000000000000000000000000000000000000000007D4FE747

```

**RTX1030**

```bash
./kk -r 31 -p 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28 -R
[+] KeyKiller v.007
[+] Search: 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28 [Public Key]
[+] Start Fri Sep 19 04:57:39 2025
[+] Random mode
[+] Range (2^31)
[+] from : 0x40000000
[+] to   : 0x7FFFFFFF
[+] GPU: GPU #0 NVIDIA GeForce GT 1030 (3x128 cores) Grid(384x256)
[+] Starting keys set in 0.03 seconds
[+] [GPU 98.95 Mkey/s][Total 2^26.58][Prob 9.4e+00%][50% in seconds][Found 0]  

[!] (Pub): 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28
[!] (WIF): Compressed:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M9SmFMSCA4jQRW
[!] (HEX): 0x000000000000000000000000000000000000000000000000000000007D4FE747

```

## Compile

```bash
make
git clone https://github.com/8891689/KeyKiller-Cuda.git
```
## Local test based on 1030

1. Address and HASH160 Mode
```bash

| GPU               | Grid      | Speed (Mkeys/s) | Notes        |
| RTX1030           | 256,256   | 56 Mkeys/s    | My test      |
```

2. Public Key Mode
```bash

| GPU               | Grid      | Speed (Mkeys/s) | Notes        |
| RTX1030           | 256,256   | 99.6 Mkeys/s    | My test      |

```
# Sponsorship
If this project has been helpful or inspiring, please consider buying me a coffee. Your support is greatly appreciated. Thank you!
```
BTC: bc1qt3nh2e6gjsfkfacnkglt5uqghzvlrr6jahyj2k
ETH: 0xD6503e5994bF46052338a9286Bc43bC1c3811Fa1
DOGE: DTszb9cPALbG9ESNJMFJt4ECqWGRCgucky
TRX: TAHUmjyzg7B3Nndv264zWYUhQ9HUmX4Xu4
```

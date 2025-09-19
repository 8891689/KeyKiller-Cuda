// Vanity.cpp
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
#include "Vanity.h"

#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Wildcard.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <string.h>
#include <math.h>
#include <algorithm> // 8891689_FIX: 包含 algorithm 以使用 std::all_of 和 std::reverse
#include <thread>
#include <atomic>
#include <ctime> 
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cctype>      // 8891689_FIX: 包含 cctype 以使用 isxdigit

using namespace std;

// 8891689_MOD: 構造函數的輸入參數名改為 targets 以反映其通用性
VanitySearch::VanitySearch(Secp256K1* secp, std::vector<std::string>& targets, int searchMode,
    bool stop, std::string outputFile, uint32_t maxFound, BITCRACK_PARAM* bc) : inputAddresses(targets)
{
	this->secp = secp;
	this->searchMode = searchMode;
	this->stopWhenFound = stop;
	this->outputFile = outputFile;
	this->numGPUs = 0;
	this->maxFound = maxFound;	
	this->searchType = -1;
	this->bc = bc;	

	rseed(static_cast<unsigned long>(time(NULL)));
	
	addresses.clear();

	ADDRESS_TABLE_ITEM t;
	t.found = true;
	t.items = NULL;
	for (int i = 0; i < 65536; i++)
		addresses.push_back(t);
	
	nbAddress = 0;
	onlyFull = true;

	for (int i = 0; i < (int)targets.size(); i++) 
	{
		ADDRESS_ITEM it;
		if (initAddress(targets[i], &it)) {
            // 正常地址處理邏輯
			bool* found = new bool;
			*found = false;
			it.found = found;
			
            address_t p = it.sAddress;
            if (addresses[p].items == NULL) {
                addresses[p].items = new vector<ADDRESS_ITEM>();
                addresses[p].found = false;
                usedAddress.push_back(p);
            }
            (*addresses[p].items).push_back(it);
			
			onlyFull &= it.isFull;
			nbAddress++;
		}
        // 8891689_MOD: 如果是公鑰模式，我們仍然需要計數以啟動搜索
        else if (this->searchType == PUBKEY) {
            nbAddress++; 
            onlyFull = false; 
        }
	}

	if (nbAddress == 0) 
	{
		fprintf(stderr, "[ERROR] VanitySearch: nothing to search !\n");
		exit(-1);
	}

	for (int i = 0; i < (int)addresses.size(); i++) 
	{
		if (addresses[i].items) 
		{
			LADDRESS lit;
			lit.sAddress = i;
			for (int j = 0; j < (int)addresses[i].items->size(); j++) 
			{
				lit.lAddresses.push_back((*addresses[i].items)[j].lAddress);
			}
			sort(lit.lAddresses.begin(), lit.lAddresses.end());
			usedAddressL.push_back(lit);
		}
	}
	
	beta.SetBase16("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
	lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
	beta2.SetBase16("851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");
	lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");
}

// 8891689_MOD: 新增時間格式化輔助函數
string VanitySearch::format_time_long(double seconds) {
    if (seconds < 1.0 || !isfinite(seconds)) return "soon";
    if (seconds < 60.0) return "seconds";
    if (seconds < 3600.0) return "minutes";
    if (seconds < 86400.0) return "hours";
    
    double years = seconds / (86400.0 * 365.25);
    stringstream ss;
    ss << scientific << setprecision(4) << years << "y";
    return ss.str();
}

bool VanitySearch::isSingularAddress(std::string pref) {

	// check is the given address contains only 1
	bool only1 = true;
	int i = 0;
	while (only1 && i < (int)pref.length()) {
		only1 = pref.data()[i] == '1';
		i++;
	}
	return only1;
}


// 8891689_MOD:  重寫 initAddress 函數以清晰地處理公鑰和地址
bool VanitySearch::initAddress(std::string& address, ADDRESS_ITEM* it) {

    // --- 步驟 1: 優先檢測是否為十六進制公鑰格式 ---
    bool isHex = address.length() > 0 && std::all_of(address.cbegin(), address.cend(), [](char c){ return std::isxdigit(c); });
    bool isCompressedPubKey = isHex && address.length() == 66 && (address.substr(0, 2) == "02" || address.substr(0, 2) == "03");
    bool isUncompressedPubKey = isHex && address.length() == 130 && address.substr(0, 2) == "04";

    if (isCompressedPubKey || isUncompressedPubKey) {
        if (searchType != -1 && searchType != PUBKEY) {
            fprintf(stdout, "[ERROR] Ignoring public key \"%s\". Cannot mix public key search (-p) with address search (-a).\n", address.c_str());
            return false;
        }
        
        searchType = PUBKEY;
        
        // 8891689_FIX: 創建一個可寫的 char* 緩存來傳遞給 SetBase16
        std::string x_hex = address.substr(2, 64);
        std::vector<char> x_writable(x_hex.begin(), x_hex.end());
        x_writable.push_back('\0');

        Int x_coord;
        x_coord.SetBase16(x_writable.data());

        if (isCompressedPubKey) {
            this->targetPubKeyParity = (address.substr(0, 2) == "03"); // 03 is odd (1), 02 is even (0)
        } else { // isUncompressedPubKey
            std::string y_hex = address.substr(66, 64);
            std::vector<char> y_writable(y_hex.begin(), y_hex.end());
            y_writable.push_back('\0');
            
            Int y_coord;
            y_coord.SetBase16(y_writable.data());
            
            Point p;
            p.x.Set(&x_coord);
            p.y.Set(&y_coord);
            p.z.SetInt32(1); // 假設 z=1
            if (!secp->EC(p)) {
                fprintf(stderr, "[ERROR] Invalid uncompressed public key provided: Point is not on the curve.\n");
                exit(-1);
            }
            this->targetPubKeyParity = p.y.IsOdd();
            fprintf(stdout, "[INFO] Uncompressed public key detected. Searching for its compressed form (0%c...).\n", this->targetPubKeyParity ? '3' : '2');
        }

        memcpy(this->targetPubKeyX, x_coord.bits64, sizeof(this->targetPubKeyX));

        return false;
    }

    // --- 步驟 2: 如果不是公鑰，則按原邏輯處理地址 ---
	std::vector<unsigned char> result;
	string dummy1 = address;
	int nbDigit = 0;
	bool wrong = false;

	if (address.length() < 2) {
		fprintf(stdout, "Ignoring address \"%s\" (too short)\n", address.c_str());
		return false;
	}

	int aType = -1;

	switch (address.data()[0]) {
	case '1':
		aType = P2PKH;
		break;
	case '3':
		aType = P2SH;
		break;
	case 'b':
	case 'B':
		std::transform(address.begin(), address.end(), address.begin(), ::tolower);
		if (strncmp(address.c_str(), "bc1q", 4) == 0)
			aType = BECH32;
		break;
	}

	if (aType == -1) {
		fprintf(stdout, "Ignoring target \"%s\" (not a valid address or compressed/uncompressed public key)\n", address.c_str());
		return false;
	}

	if (searchType == -1) searchType = aType;
	if (aType != searchType) {
		fprintf(stdout, "Ignoring address \"%s\" (P2PKH, P2SH or BECH32 allowed at once)\n", address.c_str());
		return false;
	}

	if (aType == BECH32) {

		// BECH32
		uint8_t witprog[40];
		size_t witprog_len;
		int witver;
		const char* hrp = "bc";

		int ret = segwit_addr_decode(&witver, witprog, &witprog_len, hrp, address.c_str());

		// Try to attack a full address ?
		if (ret && witprog_len == 20) {
						
			it->isFull = true;
			memcpy(it->hash160, witprog, 20);
			it->sAddress = *(address_t*)(it->hash160);
			it->lAddress = *(addressl_t*)(it->hash160);
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;

		}

		if (address.length() < 5) {
			fprintf(stdout, "Ignoring address \"%s\" (too short, length<5 )\n", address.c_str());
			return false;
		}

		if (address.length() >= 36) {
			fprintf(stdout, "Ignoring address \"%s\" (too long, length>36 )\n", address.c_str());
			return false;
		}

		uint8_t data[64];
		memset(data, 0, 64);
		size_t data_length;
		if (!bech32_decode_nocheck(data, &data_length, address.c_str() + 4)) {
			fprintf(stdout, "Ignoring address \"%s\" (Only \"023456789acdefghjklmnpqrstuvwxyz\" allowed)\n", address.c_str());
			return false;
		}
		
		it->sAddress = *(address_t*)data;		
		it->isFull = false;
		it->lAddress = 0;
		it->address = (char*)address.c_str();
		it->addressLength = (int)address.length();

		return true;
	}
	else {

		// P2PKH/P2SH
		wrong = !DecodeBase58(address, result);

		if (wrong) {
			fprintf(stdout, "Ignoring address \"%s\" (0, I, O and l not allowed)\n", address.c_str());
			return false;
		}

		// Try to attack a full address ?
		if (result.size() > 21) {
			
			it->isFull = true;
			memcpy(it->hash160, result.data() + 1, 20);
			it->sAddress = *(address_t*)(it->hash160);
			it->lAddress = *(addressl_t*)(it->hash160);
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;
		}

		// Address containing only '1'
		if (isSingularAddress(address)) {

			if (address.length() > 21) {
				fprintf(stdout, "Ignoring address \"%s\" (Too much 1)\n", address.c_str());
				return false;
			}
			
			it->isFull = false;
			it->sAddress = 0;
			it->lAddress = 0;
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;
		}

		// Search for highest hash160 16bit address (most probable)
		while (result.size() < 25) {
			DecodeBase58(dummy1, result);
			if (result.size() < 25) {
				dummy1.append("1");
				nbDigit++;
			}
		}

		if (searchType == P2SH) {
			if (result.data()[0] != 5) {
				fprintf(stdout, "Ignoring address \"%s\" (Unreachable, 31h1 to 3R2c only)\n", address.c_str());
				return false;
			}
		}

		if (result.size() != 25) {
			fprintf(stdout, "Ignoring address \"%s\" (Invalid size)\n", address.c_str());
			return false;
		}

		it->sAddress = *(address_t*)(result.data() + 1);

		dummy1.append("1");
		DecodeBase58(dummy1, result);

		if (result.size() == 25) {
			it->sAddress = *(address_t*)(result.data() + 1);
			nbDigit++;
		}
		
		it->isFull = false;
		it->lAddress = 0;
		it->address = (char*)address.c_str();
		it->addressLength = (int)address.length();

		return true;
	}
}

void VanitySearch::enumCaseUnsentiveAddress(std::string s, std::vector<std::string>& list) {

	char letter[64];
	int letterpos[64];
	int nbLetter = 0;
	int length = (int)s.length();

	for (int i = 1; i < length; i++) {
		char c = s.data()[i];
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
			letter[nbLetter] = tolower(c);
			letterpos[nbLetter] = i;
			nbLetter++;
		}
	}

	int total = 1 << nbLetter;

	for (int i = 0; i < total; i++) {

		char tmp[64];
		strcpy(tmp, s.c_str());

		for (int j = 0; j < nbLetter; j++) {
			int mask = 1 << j;
			if (mask & i) tmp[letterpos[j]] = toupper(letter[j]);
			else         tmp[letterpos[j]] = letter[j];
		}

		list.push_back(string(tmp));

	}

}

// ----------------------------------------------------------------------------

// 8891689_MOD: 修正 output 函數，增加默認保存到 found.txt 的邏輯
void VanitySearch::output(string target, string pAddr, string pAddrHex) {

#ifdef WIN64
	WaitForSingleObject(ghMutex, INFINITE);
#else
	pthread_mutex_lock(&ghMutex);
#endif

    // --- 步驟 1: 默認保存到 found.txt ---
    FILE* foundFile = fopen("found.txt", "a");
    if (foundFile == NULL) {
        fprintf(stderr, "\n[WARNING] Could not open found.txt for writing.\n");
    }

	// --- -o 參數文件處理邏輯  ---
	FILE* f = stdout;
	bool needToClose = false;

	if (outputFile.length() > 0) {
		f = fopen(outputFile.c_str(), "a");
		if (f == NULL) {
			fprintf(stderr, "Cannot open %s for writing\n", outputFile.c_str());
			f = stdout;
		}
		else {
			needToClose = true;
		}
	}

    // 手動為十六進制字符串填充前導零
    string paddedHex = pAddrHex;
    if (paddedHex.length() < 64) {
        paddedHex.insert(0, 64 - paddedHex.length(), '0');
    }

    // --- 步驟 2: 格式化輸出內容 ---
    stringstream ss;
    if (this->searchType == PUBKEY) {
        ss << "\n[!] (Pub): " << target << "\n";
        ss << "[!] (WIF): Compressed:" << pAddr << "\n";
    } else {
        ss << "\n[!] (Add): " << target << "\n";
        string wif_type = "p2pkh"; // 默認為 p2pkh (壓縮)
        if(target.length() > 0){
            if (target[0] == '3') wif_type = "p2wpkh-p2sh";
            else if (target.rfind("bc1", 0) == 0) wif_type = "p2wpkh";
        }
        ss << "[!] (WIF): " << wif_type << ":" << pAddr << "\n";
    }
    ss << "[!] (HEX): 0x" << paddedHex << "\n";

    string output_str = ss.str();

    // --- 步驟 3: 將內容同時打印到屏幕和文件 ---
    
    // 打印到屏幕 (stdout)
	fprintf(stdout, "\n%s", output_str.c_str());
    fflush(stdout);

    // 寫入到 -o 指定的文件 (暫時不需要)
	if (f != stdout) {
        fprintf(f, "%s", output_str.c_str());
        fflush(f);
    }
    
    // 寫入到默認的 found.txt
    if (foundFile != NULL) {
        fprintf(foundFile, "%s", output_str.c_str());
        fflush(foundFile);
    }

    // --- 步驟 4: 關閉文件句柄 ---
	if (needToClose) {
        fclose(f);
    }
    if (foundFile != NULL) {
        fclose(foundFile);
    }

#ifdef WIN64
	ReleaseMutex(ghMutex);
#else
	pthread_mutex_unlock(&ghMutex);
#endif
}


void VanitySearch::updateFound() {

	// Check if all addresses has been found
	// Needed only if stopWhenFound is asked
	if (stopWhenFound) 	{

		bool allFound = true;
		for (int i = 0; i < (int)usedAddress.size(); i++) {
			bool iFound = true;
			address_t p = usedAddress[i];
			if (!addresses[p].found) {
				if (addresses[p].items) {
					for (int j = 0; j < (int)addresses[p].items->size(); j++) {
						iFound &= *((*addresses[p].items)[j].found);
					}
				}
				addresses[usedAddress[i]].found = iFound;
			}
			allFound &= iFound;
		}

		endOfSearch = allFound;		
	}		
}

bool VanitySearch::checkPrivKey(string addr, Int& key, int32_t incr, int endomorphism, bool mode) {

	Int k(&key);	

	if (incr < 0) {
		k.Add((uint64_t)(-incr));
		k.Neg();
		k.Add(&secp->order);		
	}
	else {
		k.Add((uint64_t)incr);
	}

	// Endomorphisms
	switch (endomorphism) {
	case 1:
		k.ModMulK1order(&lambda);		
		break;
	case 2:
		k.ModMulK1order(&lambda2);		
		break;
	}

	// Check addresses
	Point p = secp->ComputePublicKey(&k);	

	string chkAddr = secp->GetAddress(searchType, mode, p);
	if (chkAddr != addr) {

		// Key may be the opposite one (negative zero or compressed key)
		k.Neg();
		k.Add(&secp->order);
		p = secp->ComputePublicKey(&k);
		
		string chkAddr = secp->GetAddress(searchType, mode, p);
		if (chkAddr != addr) {
			fprintf(stdout, "\nWarning, wrong private key generated !\n");
			fprintf(stdout, "  Addr :%s\n", addr.c_str());
			fprintf(stdout, "  Check:%s\n", chkAddr.c_str());
			fprintf(stdout, "  Endo:%d incr:%d comp:%d\n", endomorphism, incr, mode);
			return false;
		}

	}

	output(chkAddr, secp->GetPrivAddress(mode, k), k.GetBase16());

	return true;
}

void VanitySearch::checkAddrSSE(uint8_t* h1, uint8_t* h2, uint8_t* h3, uint8_t* h4,
	int32_t incr1, int32_t incr2, int32_t incr3, int32_t incr4,
	Int& key, int endomorphism, bool mode) {

	vector<string> addr = secp->GetAddress(searchType, mode, h1, h2, h3, h4);

	for (int i = 0; i < (int)inputAddresses.size(); i++) {

		if (Wildcard::match(addr[0].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[0], key, incr1, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[1].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[1], key, incr2, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[2].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[2], key, incr3, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[3].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[3], key, incr4, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}
	}
}

void VanitySearch::checkAddr(int prefIdx, uint8_t* hash160, Int& key, int32_t incr, int endomorphism, bool mode) {
	
	vector<ADDRESS_ITEM>* pi = addresses[prefIdx].items;	


	if (onlyFull) {

		// Full addresses
		for (int i = 0; i < (int)pi->size(); i++) {

			if (stopWhenFound && *((*pi)[i].found))
				continue;

			if (ripemd160_comp_hash((*pi)[i].hash160, hash160)) {

				// Found it !
				*((*pi)[i].found) = true;
				// You believe it ?
				if (checkPrivKey(secp->GetAddress(searchType, mode, hash160), key, incr, endomorphism, mode)) {
					nbFoundKey++;
					updateFound();
				}

			}

		}

	}
	else {
		char a[64];

		string addr = secp->GetAddress(searchType, mode, hash160);

		for (int i = 0; i < (int)pi->size(); i++) {

			if (stopWhenFound && *((*pi)[i].found))
				continue;

			strncpy(a, addr.c_str(), (*pi)[i].addressLength);
			a[(*pi)[i].addressLength] = 0;

			if (strcmp((*pi)[i].address, a) == 0) {

				// Found it !
				*((*pi)[i].found) = true;
				if (checkPrivKey(addr, key, incr, endomorphism, mode)) {
					nbFoundKey++;
					updateFound();
				}

			}

		}

	}


}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam) {
#else
void* _FindKeyGPU(void* lpParam) {
#endif
	TH_PARAM* p = (TH_PARAM*)lpParam;
	p->obj->FindKeyGPU(p);
	return 0;
}

void VanitySearch::checkAddresses(bool compressed, Int key, int i, Point p1) {

	unsigned char h0[20];
	Point pte1[1];
	Point pte2[1];

	// Point
	secp->GetHash160(searchType, compressed, p1, h0);
	address_t pr0 = *(address_t*)h0;
	if (addresses[pr0].items)
		checkAddr(pr0, h0, key, i, 0, compressed);	
}

void VanitySearch::checkAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4) {

	unsigned char h0[20];
	unsigned char h1[20];
	unsigned char h2[20];
	unsigned char h3[20];
	Point pte1[4];
	Point pte2[4];
	address_t pr0;
	address_t pr1;
	address_t pr2;
	address_t pr3;

	// Point -------------------------------------------------------------------------
	secp->GetHash160(searchType, compressed, p1, p2, p3, p4, h0, h1, h2, h3);	

	pr0 = *(address_t*)h0;
	pr1 = *(address_t*)h1;
	pr2 = *(address_t*)h2;
	pr3 = *(address_t*)h3;

	if (addresses[pr0].items)
		checkAddr(pr0, h0, key, i, 0, compressed);
	if (addresses[pr1].items)
		checkAddr(pr1, h1, key, i + 1, 0, compressed);
	if (addresses[pr2].items)
		checkAddr(pr2, h2, key, i + 2, 0, compressed);
	if (addresses[pr3].items)
		checkAddr(pr3, h3, key, i + 3, 0, compressed);	
}

void VanitySearch::getGPUStartingKeys(Int& tRangeStart, Int& tRangeEnd, int groupSize, int nbThread, Point *p, uint64_t Progress) {
		
	uint32_t grp_startkeys = nbThread/256;

	//New setting key by fixedpaul using addition on secp with batch modular inverse, super fast, multithreading not needed

	Int stepThread;
	Int numthread;

	stepThread.Set(&bc->ksFinish);
	stepThread.Sub(&bc->ksStart);
	stepThread.AddOne();
	numthread.SetInt32(nbThread);
	stepThread.Div(&numthread);

	Point Pdouble;
	Int kDouble;

	kDouble.Set(&stepThread);
	kDouble.Mult(grp_startkeys);
	Pdouble = secp->ComputePublicKey(&kDouble);

	Point P_start;
	Int kStart;

	kStart.Set(&stepThread);
	kStart.Mult(grp_startkeys / 2);
	kStart.Add(groupSize / 2 + Progress);

	

	P_start = secp->ComputePublicKey(&kStart);

	p[grp_startkeys / 2] = secp->ComputePublicKey(&tRangeStart);
	p[grp_startkeys / 2] = secp->AddDirect(p[grp_startkeys / 2], P_start);


	Int key_delta;
	Point* p_delta;
	p_delta = new Point[grp_startkeys / 2];

	key_delta.Set(&stepThread);

	

	p_delta[0] = secp->ComputePublicKey(&key_delta);
	key_delta.Add(&stepThread);
	p_delta[1] = secp->ComputePublicKey(&key_delta);

	for (size_t i = 2; i < grp_startkeys / 2; i++) {
		p_delta[i] = secp->AddDirect(p_delta[i - 1], p_delta[0]);
	}

	Int* dx;
	Int* subp;

	subp = new Int[grp_startkeys / 2 + 1];
	dx = new Int[grp_startkeys / 2 + 1];

	uint32_t j;
	//uint32_t i;

	for (size_t i = grp_startkeys / 2; i < nbThread; i += grp_startkeys) {

		double percentage = (100.0 * (double)(i + grp_startkeys / 2)) / (double)(nbThread);
		printf("Setting starting keys... [%.2f%%] \r", percentage);
		fflush(stdout);


		for (j = 0; j < grp_startkeys / 2; j++) {
			dx[j].ModSub(&p_delta[j].x, &p[i].x);
		}
		dx[grp_startkeys / 2].ModSub(&Pdouble.x, &p[i].x);

		Int newValue;
		Int inverse;

		subp[0].Set(&dx[0]);
		for (size_t j = 1; j < grp_startkeys / 2 + 1; j++) {
			subp[j].ModMulK1(&subp[j - 1], &dx[j]);
		}

		// Do the inversion
		inverse.Set(&subp[grp_startkeys / 2]);
		inverse.ModInv();

		for (j = grp_startkeys / 2; j > 0; j--) {
			newValue.ModMulK1(&subp[j - 1], &inverse);
			inverse.ModMulK1(&dx[j]);
			dx[j].Set(&newValue);
		}

		dx[0].Set(&inverse);

		Int _s;
		Int _p;
		Int dy;
		Int syn;
		syn.Set(&p[i].y);
		syn.ModNeg();



		for (j = 0; j < grp_startkeys / 2 - 1; j++) {

			dy.ModSub(&p_delta[j].y, &p[i].y);
			_s.ModMulK1(&dy, &dx[j]);

			_p.ModSquareK1(&_s);

			p[i + j + 1].x.ModSub(&_p, &p[i].x);
			p[i + j + 1].x.ModSub(&p_delta[j].x);

			p[i + j + 1].y.ModSub(&p_delta[j].x, &p[i + j + 1].x);
			p[i + j + 1].y.ModMulK1(&_s);
			p[i + j + 1].y.ModSub(&p_delta[j].y);

			dy.ModSub(&syn, &p_delta[j].y);
			_s.ModMulK1(&dy, &dx[j]);

			_p.ModSquareK1(&_s);

			p[i - j - 1].x.ModSub(&_p, &p[i].x);
			p[i - j - 1].x.ModSub(&p_delta[j].x);

			p[i - j - 1].y.ModSub(&p[i - j - 1].x, &p_delta[j].x);
			p[i - j - 1].y.ModMulK1(&_s);
			p[i - j - 1].y.ModSub(&p_delta[j].y, &p[i - j - 1].y);
		}

		dy.ModSub(&syn, &p_delta[j].y);
		_s.ModMulK1(&dy, &dx[j]);

		_p.ModSquareK1(&_s);


		p[i - j - 1].x.ModSub(&_p, &p[i].x);
		p[i - j - 1].x.ModSub(&p_delta[j].x);

		p[i - j - 1].y.ModSub(&p[i - j - 1].x, &p_delta[j].x);
		p[i - j - 1].y.ModMulK1(&_s);
		p[i - j - 1].y.ModSub(&p_delta[j].y, &p[i - j - 1].y);

		if (i + grp_startkeys < nbThread) {

			dy.ModSub(&Pdouble.y, &p[i].y);
			_s.ModMulK1(&dy, &dx[grp_startkeys / 2]);

			_p.ModSquareK1(&_s);

			p[i + grp_startkeys].x.ModSub(&_p, &p[i].x);
			p[i + grp_startkeys].x.ModSub(&Pdouble.x);

			p[i + grp_startkeys].y.ModSub(&Pdouble.x, &p[i + grp_startkeys].x);
			p[i + grp_startkeys].y.ModMulK1(&_s);
			p[i + grp_startkeys].y.ModSub(&Pdouble.y);
		}
	}

	delete[] subp;
	delete[] dx;
	delete[] p_delta;
}


// 8891689_MOD: 修正并重新實現集成了公钥搜索模式的 FindKeyGPU 函數

void VanitySearch::FindKeyGPU(TH_PARAM* ph) {

    bool ok = true;
    double t0 = 0.0, ttot = 0.0;
    uint64_t keys_n = 0;
    vector<ITEM> found;
    
    Int total_keyspace;
    total_keyspace.Sub(&bc->ksFinish, &bc->ksStart);
    total_keyspace.AddOne();

    ph->hasStarted = true;
    endOfSearch = false; 

    // 主循環，控制整個搜索過程，包括暫停和恢復
    while (!endOfSearch) {

        // 如果處於暫停狀態，就在這裡等待，直到用戶按 'p' 恢復
        if (Pause) {
            if (!Paused) { // 只打印一次暫停信息
                printf("\n[+] Paused. Press 'p' to resume or Ctrl+C to exit.\r");
                fflush(stdout);
                Paused = true;
            }
            Timer::SleepMillis(100); // 避免 CPU 100% 空轉
            continue; // 跳過本次循環，繼續檢查 Pause 狀態
        }

        // 如果執行到這裡，說明 !Pause，即程序需要運行或恢復運行

        // ====================================================================
        // GPU 初始化塊：在搜索開始時和每次從暫停恢復時都會執行
        // ====================================================================
        if (Paused) { // 如果 Paused 為 true，說明我們是從暫停狀態恢復的
            printf("\n[+] Resuming search...\n");
            fflush(stdout);
        }

        GPUEngine g(ph->gpuId, maxFound);
        // 檢查 GPUEngine 是否成功初始化 
        if (!g.IsInitialised()) {
            fprintf(stderr, "\n[ERROR] Failed to initialize GPU Engine. Exiting thread.\n");
            break; // 退出主循環
        }
        
        printf("[+] GPU: %s\n", g.deviceName.c_str());
        fflush(stdout);
        
        if (this->searchType == PUBKEY) {
            g.SetTargetPublicKey(this->targetPubKeyX, this->targetPubKeyParity);
        } else {
            g.SetSearchType(searchType);
            g.SetAddress(usedAddressL, nbAddress);
        }

        int numThreadsGPU = g.GetNbThread();
        int STEP_SIZE = g.GetStepSize();
        Point* publicKeys = new Point[numThreadsGPU];
        
        Int stepThread;
        Int num_threads_int((uint64_t)numThreadsGPU);
        stepThread.Set(&total_keyspace);
        stepThread.Div(&num_threads_int);

        double setup_t0 = Timer::get_tick();
        getGPUStartingKeys(bc->ksStart, bc->ksFinish, g.GetGroupSize(), numThreadsGPU, publicKeys, (uint64_t)(1ULL * idxcount * STEP_SIZE));
        ok = g.SetKeys(publicKeys);
        delete[] publicKeys;
        double setup_time = Timer::get_tick() - setup_t0;
        printf("[+] Starting keys set in %.2f seconds\n", setup_time);
        fflush(stdout);

        Paused = false; // 重置 Paused 標誌，表示現在正在運行
        t0 = Timer::get_tick(); // 重置當前運行段的計時器

        // ====================================================================
        // 核心計算循環：只要不暫停，就一直執行
        // ====================================================================
        while (ok && !endOfSearch && !Pause) {
            ok = g.Launch(found, true);
            idxcount += 1;

            ttot = Timer::get_tick() - t0 + t_Paused;

            if (backupMode && !randomMode && idxcount % 60 == 0) {
                saveBackup(idxcount, ttot, ph->gpuId);
            }

            Int keys_done;
            keys_done.SetInt32(idxcount);
            keys_done.Mult(numThreadsGPU);
            keys_done.Mult(STEP_SIZE);
            
            if (!randomMode && keys_done.IsGreaterOrEqual(&total_keyspace)) {
                endOfSearch = true;
                printf("\n[+] Search range completed.\r");
                fflush(stdout);
            }

            Int keycount;
            keycount.SetInt32(idxcount - 1);
            keycount.Mult(STEP_SIZE);

            for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
                ITEM it = found[i];
                Int part_key;
                part_key.Set(&stepThread);
                part_key.Mult(it.thId);
    
                Int privkey;
                privkey.Set(&bc->ksStart);
                privkey.Add(&part_key);
                privkey.Add(&keycount);
            
                if (this->searchType == PUBKEY) {
                    Int k(&privkey);
                    if (it.incr < 0) {
                        k.Add((uint64_t)(-it.incr));
                        k.Neg();
                        k.Add(&secp->order);
                    } else {
                        k.Add((uint64_t)it.incr);
                    }
                    output(this->inputAddresses[0], secp->GetPrivAddress(true, k), k.GetBase16());
                    nbFoundKey++;
                    updateFound();
                } else {
                    checkAddr(*(address_t*)(it.hash), it.hash, privkey, it.incr, it.endo, it.mode);
                }
            }

            keys_n = 1ULL * STEP_SIZE * numThreadsGPU * idxcount;
            PrintStats(keys_n, ttot, total_keyspace);
        } // 內層 while 結束 (因為 Pause 或 endOfSearch)

        // 保存本次運行段的時間
        // GPUEngine 'g' 在這裡會自動銷毀，釋放資源，為下一次暫停做準備
        // 這是 C++ RAII (資源獲取即初始化) 的美妙之處
        t_Paused += (Timer::get_tick() - t0);

    } // 外層 while 結束

    if (backupMode && !randomMode) {
        double final_ttot = Timer::get_tick() - t0 + t_Paused;
        saveBackup(idxcount, final_ttot, ph->gpuId);
    }
    
    ph->isRunning = false;
}

// 8891689_MOD: 重寫 PrintStats 函數，增加退出判斷
void VanitySearch::PrintStats(uint64_t keys_n, double ttot, const Int& total_keyspace) {
    // 8891689_FIX: 如果程序正在關閉，則不打印任何狀態信息
    if (endOfSearch) {
        return;
    }

    if (ttot < 0.1) return;

    double speed = (keys_n > 0) ? static_cast<double>(keys_n) / ttot / 1000000.0 : 0.0;
    double log_keys = (keys_n > 0) ? log2(static_cast<double>(keys_n)) : 0.0;
    
    double total_space_double = total_keyspace.ToDouble();
    double prob = (keys_n > 0) ? (static_cast<double>(keys_n) / total_space_double) : 0.0;
    
    double keys_per_sec = speed * 1000000.0;
    double time_to_50_percent = (keys_per_sec > 0) ? (0.5 * total_space_double) / keys_per_sec : std::numeric_limits<double>::infinity();

    string time_str_50 = format_time_long(time_to_50_percent);

    if (randomMode) {
         printf("[+] [GPU %.2f Mkey/s][Total 2^%.2f][Prob %.1e%%][50%% in %s] \r",
               speed, log_keys, prob * 100.0, time_str_50.c_str(), nbFoundKey);
    } else {
        // 8891689_FIX: 為了美觀，在百分比後面加個空格
        printf("[+] [GPU %.2f Mkey/s][Total 2^%.2f][Prob %.2f%%][50%% in %s] \r",
               speed, log_keys, prob * 100.0, time_str_50.c_str(), nbFoundKey);
    }

    fflush(stdout);
}


void VanitySearch::saveBackup(int idxcount, double t_Paused, int gpuid) {
	std::string filename = "schedule_gpu" + std::to_string(gpuid) + ".dat";
	std::ofstream outFile(filename, std::ios::binary);
	if (outFile) {
		outFile.write(reinterpret_cast<const char*>(&idxcount), sizeof(idxcount));
		outFile.write(reinterpret_cast<const char*>(&t_Paused), sizeof(t_Paused));
		outFile.close();
	}
	else {
		std::cerr << "Error opening file for writing: " << filename << "\n";
	}
}

bool VanitySearch::isAlive(TH_PARAM * p) {

	bool isAlive = true;
	int total = numGPUs;
	for (int i = 0; i < total; i++)
		isAlive = isAlive && p[i].isRunning;

	return isAlive;
}

bool VanitySearch::hasStarted(TH_PARAM * p) {

	bool hasStarted = true;
	int total = numGPUs;
	for (int i = 0; i < total; i++)
		hasStarted = hasStarted && p[i].hasStarted;

	return hasStarted;
}

uint64_t VanitySearch::getGPUCount() {

	uint64_t count = 0;
	for (int i = 0; i < numGPUs; i++) {
		count += counters[i];
	}
	return count;
}

void VanitySearch::saveProgress(TH_PARAM* p, Int& lastSaveKey, BITCRACK_PARAM* bc) {

	Int lowerKey;
	lowerKey.Set(&p[0].THnextKey);

	int total = numGPUs;
	for (int i = 0; i < total; i++) {
		if (p[i].THnextKey.IsLower(&lowerKey))
			lowerKey.Set(&p[i].THnextKey);
	}

	if (lowerKey.IsLowerOrEqual(&lastSaveKey)) return;
	lastSaveKey.Set(&lowerKey);
}

void VanitySearch::Search(std::vector<int> gpuId, std::vector<int> gridSize) {
	endOfSearch = false;
	numGPUs = (int)gpuId.size();
	nbFoundKey = 0;

	memset(counters, 0, sizeof(counters));	

	TH_PARAM* params = (TH_PARAM*)malloc(numGPUs * sizeof(TH_PARAM));
	memset(params, 0, numGPUs * sizeof(TH_PARAM));
	
	// 8891689_FIX: 確保動態分配的線程數組被釋放
	std::thread* threads = new std::thread[numGPUs];

#ifdef WIN64
	ghMutex = CreateMutex(NULL, FALSE, NULL);
	mutex = CreateMutex(NULL, FALSE, NULL);
#else
	ghMutex = PTHREAD_MUTEX_INITIALIZER;
	mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

	// 啟動 GPU 線程
	for (int i = 0; i < numGPUs; i++) {
		params[i].obj = this;
		params[i].threadId = i;
		params[i].isRunning = true;
		params[i].gpuId = gpuId[i];
		// gridSize 可以在這裡設置，如果需要的話
		params[i].gridSizeX = gridSize[i*2];
		params[i].gridSizeY = gridSize[i*2+1];
		params[i].THnextKey.Set(&bc->ksNext);
		
		threads[i] = std::thread(_FindKeyGPU, params + i);
	}

    // 等待所有線程啟動
	while (!hasStarted(params)) {
		Timer::SleepMillis(500);
	}

    // 主線程在這裡等待，直到 endOfSearch 被設置為 true
	while (!endOfSearch) {
		Timer::SleepMillis(100);
	}

    // 8891689_FIX: 這是解決問題的關鍵！
    // 等待所有 GPU 線程執行完它們的清理工作（包括最後的 saveBackup）
    for (int i = 0; i < numGPUs; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
	
	if (params != nullptr) {
		free(params);
	}
    // 8891689_FIX: 釋放線程數組內存
    delete[] threads;
}

string VanitySearch::GetHex(vector<unsigned char> &buffer) {

	string ret;

	char tmp[128];
	for (int i = 0; i < (int)buffer.size(); i++) {
		sprintf(tmp, "%02hhX", buffer[i]);
		ret.append(tmp);
	}

	return ret;
}

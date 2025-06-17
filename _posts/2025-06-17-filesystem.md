---
title:  "[Forensic] File System"
excerpt: "File System"

categories:
  - Forensic
tags:
  - Forensic
toc: true
toc_sticky: true
last_modified_at: 2023-10-18T08:06:00-05:00
---

# File System

## MBR & VBR 

HxD를 통해 MBR와 VBR의 구조를 살펴보자. `도구->디스크 열기`를 통해 조사할 물리 디스크를 선택한다. 

```
Offset(h)   00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F

0000000000  33 C0 8E D0 BC 00 7C 8E C0 8E D8 BE 00 7C BF 00  3ÀŽÐ¼.|ŽÀŽØ¾.|¿.
0000000010  06 B9 00 02 FC F3 A4 50 68 1C 06 CB FB B9 04 00  .¹..üó¤Ph..Ëû¹..
0000000020  BD BE 07 80 7E 00 00 7C 0B 0F 85 0E 01 83 C5 10  ½¾.€~..|..…..ƒÅ.
0000000030  E2 F1 CD 18 88 56 00 55 C6 46 11 05 C6 46 10 00  âñÍ.ˆV.UÆF..ÆF..
0000000040  B4 41 BB AA 55 CD 13 5D 72 0F 81 FB 55 AA 75 09  ´A»ªUÍ.]r..ûUªu.
0000000050  F7 C1 01 00 74 03 FE 46 10 66 60 80 7E 10 00 74  ÷Á..t.þF.f`€~..t
0000000060  26 66 68 00 00 00 00 66 FF 76 08 68 00 00 68 00  &fh....fÿv.h..h.
0000000070  7C 68 01 00 68 10 00 B4 42 8A 56 00 8B F4 CD 13  |h..h..´BŠV.‹ôÍ.
0000000080  9F 83 C4 10 9E EB 14 B8 01 02 BB 00 7C 8A 56 00  ŸƒÄ.žë.¸..».|ŠV.
0000000090  8A 76 01 8A 4E 02 8A 6E 03 CD 13 66 61 73 1C FE  Šv.ŠN.Šn.Í.fas.þ
00000000A0  4E 11 75 0C 80 7E 00 80 0F 84 8A 00 B2 80 EB 84  N.u.€~.€.„Š.²€ë„
00000000B0  55 32 E4 8A 56 00 CD 13 5D EB 9E 81 3E FE 7D 55  U2äŠV.Í.]ëž.>þ}U
00000000C0  AA 75 6E FF 76 00 E8 8D 00 75 17 FA B0 D1 E6 64  ªunÿv.è..u.ú°Ñæd
00000000D0  E8 83 00 B0 DF E6 60 E8 7C 00 B0 FF E6 64 E8 75  èƒ.°ßæ`è|.°ÿædèu
00000000E0  00 FB B8 00 BB CD 1A 66 23 C0 75 3B 66 81 FB 54  .û¸.»Í.f#Àu;f.ûT
00000000F0  43 50 41 75 32 81 F9 02 01 72 2C 66 68 07 BB 00  CPAu2.ù..r,fh.».
0000000100  00 66 68 00 02 00 00 66 68 08 00 00 00 66 53 66  .fh....fh....fSf
0000000110  53 66 55 66 68 00 00 00 00 66 68 00 7C 00 00 66  SfUfh....fh.|..f
0000000120  61 68 00 00 07 CD 1A 5A 32 F6 EA 00 7C 00 00 CD  ah...Í.Z2öê.|..Í
0000000130  18 A0 B7 07 EB 08 A0 B6 07 EB 03 A0 B5 07 32 E4  . ·.ë. ¶.ë. µ.2ä
0000000140  05 00 07 8B F0 AC 3C 00 74 09 BB 07 00 B4 0E CD  ...‹ð¬<.t.»..´.Í
0000000150  10 EB F2 F4 EB FD 2B C9 E4 64 EB 00 24 02 E0 F8  .ëòôëý+Éädë.$.àø
0000000160  24 02 C3 49 6E 76 61 6C 69 64 20 70 61 72 74 69  $.ÃInvalid parti
0000000170  74 69 6F 6E 20 74 61 62 6C 65 00 45 72 72 6F 72  tion table.Error
0000000180  20 6C 6F 61 64 69 6E 67 20 6F 70 65 72 61 74 69   loading operati
0000000190  6E 67 20 73 79 73 74 65 6D 00 4D 69 73 73 69 6E  ng system.Missin
00000001A0  67 20 6F 70 65 72 61 74 69 6E 67 20 73 79 73 74  g operating syst
00000001B0  65 6D 00 00 00 63 7B 9A EB E3 EB E3 00 00 80 20  em...c{šëãëã..€ 
00000001C0  21 00 07 FE FF FF 00 08 00 00 00 00 6A 18 00 FE  !..þÿÿ......j..þ
00000001D0  FF FF 07 FE FF FF 00 08 6A 18 00 50 06 5C 00 00  ÿÿ.þÿÿ..j..P.\..
00000001E0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
00000001F0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 55 AA  ..............Uª
```

섹터 0을 보면 위와 같다. 오프셋 기준 `0x0 ~ 0x1BD`까지가 부트 코드, `0x1BE ~ 0x1FD`에는 파티션 테이블, 마지막 2byte에는 `0x55 0xAA`로 시그니처 값이 있다.  

1. Boot Code (0x0 ~ 0x1BD)
2. Partition Table Entry#1 (0x1BE ~ 0x1CD)
3. Partition Table Entry#2 (0x1CE ~ 0x1DD)
4. Partition Table Entry#3 (0x1DE ~ 0x1ED)
5. Partition Table Entry#4 (0x1EE ~ 0x1FD)
6. Signature (0x1FE ~ 0x1FF)

각각의 파티션 테이블 엔트리는 16바이트이며, 하나의 엔트리는 6가지 항목으로 나누어져 있다. 

1. Boot Flag (0x00 ~ 0x00): 0x00(cannot boot), 0x80(bootable)
2. Starting CHS Address (0x01 ~ 0x03)
3. Partition Type(0x04 ~ 0x04): 0x07=NTFS, 0x83=Linux ext...
4. Ending CHS Address(0x05 ~ 0x07)
5. Starting LBA (0x08 ~ 0x0B)
6. Number of Sectors in Partition(0x0C ~ 0x0F)

CHS 주소는 Cylinder-Head-Sector 방식의 주소로 요즘에는 거의 사용하지 않는다. LBA는 해당 파티션이 시작하는 섹터를 LBA 주소 기준으로 알려준다. LBA 방식은 저장장치 내 모든 섹터들에 대해 순서대로 숫자를 지정해 주소를 계산하는 방식으로 일반적으로 사용하는 주소 지정 방식이다. 

Partition Table Entry#1을 기준으로 해석해보면 다음과 같다. 

`80 20 21 00 07 FE FF FF 00 08 00 00 00 00 6A 18`

1. Boot Code 0x80: 부팅 가능
2. Starting CHS Address 0x002120
3. Partition Type 0x07: NTFS
4. Ending CHS Address 0xFFFFFE
5. Starting LBA 0x00000800
6. Number of Sectors in Partition 0x186A0000

Starting LBA가 0x00000800 섹터이다. 섹터는 512바이트(0x200)이므로 오프셋을 계산해보면 `0x800 * 0x200 = 0x100000`이다. 해당 오프셋을 조사해보면 2048(0x800) 섹터에 해당 파티션의 VBR이 위치하는 것을 확인할 수 있다. 

```
Offset(h)   00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F

0000100000  EB 52 90 4E 54 46 53 20 20 20 20 00 02 08 00 00  ëR.NTFS    .....
0000100010  00 00 00 00 00 F8 00 00 3F 00 FF 00 00 08 00 00  .....ø..?.ÿ.....
0000100020  00 00 00 00 80 00 80 00 FF FF 69 18 00 00 00 00  ....€.€.ÿÿi.....
0000100030  00 00 0C 00 00 00 00 00 02 00 00 00 00 00 00 00  ................
0000100040  F6 00 00 00 01 00 00 00 60 E7 B5 9E 0C B6 9E D4  ö.......`çµž.¶žÔ
0000100050  00 00 00 00 FA 33 C0 8E D0 BC 00 7C FB 68 C0 07  ....ú3ÀŽÐ¼.|ûhÀ.
```




; generated: jmp stubs forwarding dxgi exports to the system dxgi.dll (g_real filled in DllMain)
EXTERN g_real:QWORD
_TEXT SEGMENT
Fwd0 PROC
    jmp QWORD PTR [g_real + 0]
Fwd0 ENDP
Fwd1 PROC
    jmp QWORD PTR [g_real + 8]
Fwd1 ENDP
Fwd2 PROC
    jmp QWORD PTR [g_real + 16]
Fwd2 ENDP
Fwd6 PROC
    jmp QWORD PTR [g_real + 48]
Fwd6 ENDP
Fwd7 PROC
    jmp QWORD PTR [g_real + 56]
Fwd7 ENDP
Fwd8 PROC
    jmp QWORD PTR [g_real + 64]
Fwd8 ENDP
Fwd9 PROC
    jmp QWORD PTR [g_real + 72]
Fwd9 ENDP
Fwd10 PROC
    jmp QWORD PTR [g_real + 80]
Fwd10 ENDP
Fwd11 PROC
    jmp QWORD PTR [g_real + 88]
Fwd11 ENDP
Fwd12 PROC
    jmp QWORD PTR [g_real + 96]
Fwd12 ENDP
Fwd13 PROC
    jmp QWORD PTR [g_real + 104]
Fwd13 ENDP
Fwd14 PROC
    jmp QWORD PTR [g_real + 112]
Fwd14 ENDP
Fwd15 PROC
    jmp QWORD PTR [g_real + 120]
Fwd15 ENDP
Fwd16 PROC
    jmp QWORD PTR [g_real + 128]
Fwd16 ENDP
Fwd17 PROC
    jmp QWORD PTR [g_real + 136]
Fwd17 ENDP
Fwd18 PROC
    jmp QWORD PTR [g_real + 144]
Fwd18 ENDP
Fwd19 PROC
    jmp QWORD PTR [g_real + 152]
Fwd19 ENDP
_TEXT ENDS
END

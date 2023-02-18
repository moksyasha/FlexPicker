MODULE MainModule
    
VAR socketdev serverSocket;
VAR socketdev clientSocket;

!CONST hand_params DUMMY_HP := [0,0,0,0];
!CONST orient DUMMY_ROT := [1,0,0,0];
!CONST confdata DUMMY_ROBCONF := [0,-1,0,0];
!CONST extjoint DUMMY_EXTAX := [9E+09,9E+09,9E+09,9E+09,9E+09,9E+09];
!CONST pos DUMMY_POS := [0,0,0];
!CONST robtarget DUMMY_ROBT := [DUMMY_POS,DUMMY_ROT,DUMMY_ROBCONF,DUMMY_EXTAX];
CONST robtarget ROBT_DEFAULT := [[0,0,0],[0,1,0,0],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
CONST speeddata SD_DEFAULT := v200;
CONST zonedata ZD_DEFAULT := z0;

FUNC string Receive()
    VAR string msg;
        
    SocketReceive clientSocket, \Str:=msg, \Time:=WAIT_MAX;
    SocketSend clientSocket, \Str:=msg + " got!";
        
    RETURN msg;
ENDFUNC


PROC Send(string msg)
    !VAR string recv_msg;
        
    SocketSend clientSocket, \Str:=msg;
    !SocketReceive clientSocket, \Str:=recv_msg, \ReadNoOfBytes:=3;
    
    !IF recv_msg="ack" THEN
    !    RETURN TRUE;
    !ELSE
    !    RETURN FALSE;
    !ENDIF
ENDPROC

    
PROC main()
    
    VAR bool check := FALSE;
    
    VAR string command;
    VAR string receive_cmd;
    VAR string tmp_str;
    
    VAR num found;
    VAR num prev_found;
    
    VAR robtarget new_point := ROBT_DEFAULT;
    
    VAR num x;
    VAR num y;
    VAR num z;
    
    MoveJ ROBT_DEFAULT,SD_DEFAULT,ZD_DEFAULT,tool0\WObj:=main_obj;
    
    SocketCreate serverSocket;
    SocketBind serverSocket, "127.0.0.1", 1488;
    SocketListen serverSocket;
    SocketAccept serverSocket, clientSocket, \Time:=WAIT_MAX;
    
    
    WHILE command<>"exit" DO
        receive_cmd:= Receive();
        !find command before SPACE
        found := StrFind(receive_cmd,1,STR_WHITE);
        
        IF found = (StrLen(receive_cmd) + 1) THEN
            Send("wrong cmd!");
        ELSE
            prev_found := found;
            command := StrPart(receive_cmd, 1, found);
        ENDIF
        
        !exmpl: MJ 12.123 -12.322 23.232 
        IF command = "MJ" THEN
            
            !find x coordinate
            found := StrFind(receive_cmd, prev_found+1, STR_WHITE);
            tmp_str := StrPart(receive_cmd, prev_found+1, found);
            check := StrToVal(tmp_str, x);
            
            !find y coordinate
            found := StrFind(receive_cmd, prev_found+1, STR_WHITE);
            tmp_str := StrPart(receive_cmd, prev_found+1, found);
            check := StrToVal(tmp_str, y);
            
            !find z coordinate
            found := StrFind(receive_cmd, prev_found+1, STR_WHITE);
            tmp_str := StrPart(receive_cmd, prev_found+1, found);
            check := StrToVal(tmp_str, z);
            
        ENDIF
        
        IF check THEN
            Send("cmd accepted");
            new_point.trans.x := x;
            new_point.trans.y := y;
            new_point.trans.z := z;
            MoveJ new_point,SD_DEFAULT,ZD_DEFAULT,tool0\WObj:=wobj0;
        ELSE
            Send("wrong cmd!");
        ENDIF
        
        check:=FALSE;
        x:=0;
        y:=0;
        z:=0;
        found:=0;
        prev_found:=0;
        
    ENDWHILE
    
    SocketClose clientSocket;
    SocketClose serverSocket;
    stop;
    
    ERROR
    IF ERRNO=ERR_SOCK_TIMEOUT THEN
        RETRY;
    ELSEIF ERRNO=ERR_SOCK_CLOSED THEN
        SocketClose clientSocket;
        SocketClose serverSocket;
        SocketCreate serverSocket;
        SocketBind serverSocket, "127.0.0.1", 1488;
        SocketListen serverSocket;
        SocketAccept serverSocket, clientSocket, \Time:=WAIT_MAX;
        RETRY;
    ELSE
        SocketClose clientSocket;
        SocketClose serverSocket;
        stop;
    ENDIF
    
ENDPROC

ENDMODULE
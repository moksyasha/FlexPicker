MODULE simple_socket_util
    
    RECORD status
        bool RobotLoopRunning;
        bool CmdLoopRunning;
        bool StatusLoopRunning;
        num n_cur_cmd;
        num n_last_cmd;
    ENDRECORD
    
    RECORD hand_params
        num hand_height;
        num items_height;
        num hand_lower_time;
        num hand_raise_time;
    ENDRECORD
    
    RECORD cmddata
        string type;
        num target_wobj_num;
        robtarget target_robt1;
        robtarget target_robt2;
        speeddata sd;
        zonedata zd;
        hand_params hp;
        num time;
    ENDRECORD
    
    RECORD basket_pos
        num item_type;
        pos trans;
    ENDRECORD
    
    RECORD catch_cmd
        num item_type;
        num time;
        pos trans;
    ENDRECORD
    
    CONST hand_params DUMMY_HP := [0,0,0,0];
    CONST orient DUMMY_ROT := [1,0,0,0];
    CONST confdata DUMMY_ROBCONF := [0,-1,0,0];
    CONST extjoint DUMMY_EXTAX := [9E+09,9E+09,9E+09,9E+09,9E+09,9E+09];
    CONST pos DUMMY_POS := [0,0,0];
    CONST robtarget DUMMY_ROBT := [DUMMY_POS,DUMMY_ROT,DUMMY_ROBCONF,DUMMY_EXTAX];
    CONST speeddata SD_DEFAULT := v200;
    CONST zonedata ZD_DEFAULT := z0;
    CONST num n_max_cmd := 3;
    
    FUNC string Receive()
        VAR string msg;
        
        SocketReceive clientSocket, \Str:=msg, \Time:=WAIT_MAX;
        SocketSend clientSocket, \Str:="ack";
        
        RETURN msg;
    ENDFUNC
    
    FUNC bool Send(string msg)
        VAR string recv_msg;
        
        SocketSend clientSocket, \Str:=msg;
        SocketReceive clientSocket, \Str:=recv_msg, \ReadNoOfBytes:=3;
        
        IF recv_msg="ack" THEN
            RETURN TRUE;
        ELSE
            RETURN FALSE;
        ENDIF
    ENDFUNC
    
ENDMODULE
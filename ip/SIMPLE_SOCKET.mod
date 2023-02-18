MODULE simple_socket
    
    PERS cmddata cmd_queue{n_max_cmd};
    VAR pos drop_pos{10};
    PERS bool lock{n_max_cmd};
    VAR cmddata new_cmd;

    VAR socketdev serverSocket;
    VAR socketdev clientSocket;
    
    ! IVAN
    VAR socketstatus state_connekt_socket;
    ! Ivan
    
    VAR num port := 5000;
    VAR string recv_cmd;
    VAR bool ok;
    PERS status s := [TRUE, TRUE, TRUE, 10, 10];
    VAR clock clk1;
    VAR num start_time;
    
    VAR string d := ",";
    VAR string l := "[";
    VAR string r := "]";
    
    FUNC bool AddNewCmd()
        VAR num n_new_cmd;
        n_new_cmd := s.n_last_cmd+1;
        IF n_new_cmd=n_max_cmd+1 THEN
            n_new_cmd := 1;
        ENDIF
        
        WaitUntil lock{n_new_cmd}=FALSE;
        lock{n_new_cmd} := TRUE;
        cmd_queue{n_new_cmd} := new_cmd;
        lock{n_new_cmd} := FALSE;
        s.n_last_cmd := n_new_cmd;
        
        RETURN TRUE;
    ENDFUNC
    
    PROC RestartTimer()
        clkReset clk1;
        ClkStart clk1;
    ENDPROC
    
    FUNC num GetCurTime()
        RETURN ClkRead(clk1) + start_time;
    ENDFUNC
    
    PROC StopTimer()
        ClkStop clk1;
    ENDPROC
    
    PROC main()
        VAR string cmd;
        VAR string tmp_str;
        VAR basket_pos tmp_bp;
        VAR hand_params tmp_hp;
        VAR catch_cmd tmp_catch;
        VAR num pos01;
        VAR num pos02;
        VAR num pos11;
        VAR num pos12;
        VAR num pos21;
        VAR num pos22;
        
        VAR num ct;
        VAR num object_counter := 0;
        
        SocketClose clientSocket;
        SocketClose serverSocket;
        
        SocketCreate serverSocket;
        SocketBind serverSocket, "192.168.0.108", port;!!!
        SocketListen serverSocket;
        SocketAccept serverSocket, clientSocket, \Time:=WAIT_MAX;
        
        RestartTimer;
        TPWrite "Cmd Start Time = ", \Num:=GetCurTime();
        
        recv_cmd := "";
        
        new_cmd := ["None", 0, DUMMY_ROBT, DUMMY_ROBT, SD_DEFAULT, ZD_DEFAULT, DUMMY_HP, 0];
                
        TCPCmdLoopPart:
        s.CmdLoopRunning := TRUE;
        WHILE cmd<>"EXIT" DO
            
            ! Ivan
            !state_connekt_socket := SocketGetStatus(clientSocket);
            !WHILE (state_connekt_socket <> SOCKET_CONNECTED) OR (state_connekt_socket <> SOCKET_BOUND) OR (state_connekt_socket <> SOCKET_LISTENING) DO
            !   SocketAccept serverSocket, clientSocket, \Time:=WAIT_MAX;
            !    TPWrite "connect=? ", \Num:=state_connekt_socket;                
            !    state_connekt_socket := SocketGetStatus(clientSocket);
            !ENDWHILE
            ! Ivan
            
            recv_cmd := Receive();
            pos01 := 1;
            pos02 := StrFind(recv_cmd, pos01+1, d);
            cmd := StrPart(recv_cmd, pos01+1, pos02-pos01-1);
            IF cmd="SetV" OR cmd="SetZ" OR cmd="SetTargetWObjNum" OR cmd="SetBasket" OR cmd="SetHandParams" THEN
                IF cmd="SetV" THEN
                    pos11 := StrFind(recv_cmd, pos02+1, l);
                    pos12 := StrFind(recv_cmd, pos11+1, r);
                    tmp_str := StrPart(recv_cmd, pos11, pos12-pos11+1);
                    ok := StrToVal(tmp_str, new_cmd.sd);
                ELSEIF cmd="SetZ" THEN
                    pos11 := StrFind(recv_cmd, pos02+1, l);
                    pos12 := StrFind(recv_cmd, pos11+1, r);
                    tmp_str := StrPart(recv_cmd, pos11, pos12-pos11+1);
                    ok := StrToVal(tmp_str, new_cmd.zd);
                ELSEIF cmd="SetTargetWObjNum" THEN
                    pos11 := StrFind(recv_cmd, pos02+1, l);
                    pos12 := StrFind(recv_cmd, pos11+1, r);
                    tmp_str := StrPart(recv_cmd, pos11+1, pos12-pos11-1);
                    ok := StrToVal(tmp_str, new_cmd.target_wobj_num);
                ELSEIF cmd="SetBasket" THEN
                    pos11 := StrFind(recv_cmd, pos02+1, l);
                    pos12 := StrFind(recv_cmd, pos11+1, r);
                    pos12 := StrFind(recv_cmd, pos12+1, r);
                    tmp_str := StrPart(recv_cmd, pos11, pos12-pos11+1);
                    ok := StrToVal(tmp_str, tmp_bp);
                    drop_pos{tmp_bp.item_type} := tmp_bp.trans;
                ELSEIF cmd="SetHandParams" THEN
                    pos11 := StrFind(recv_cmd, pos02+1, l);
                    pos12 := StrFind(recv_cmd, pos11+1, r);
                    pos12 := StrFind(recv_cmd, pos12+1, r);
                    tmp_str := StrPart(recv_cmd, pos11, pos12-pos11);
                    ok := StrToVal(tmp_str, tmp_hp);
                    new_cmd.hp := tmp_hp;
                    new_cmd.hp.hand_lower_time := tmp_hp.hand_lower_time/1000;
                    new_cmd.hp.hand_raise_time := tmp_hp.hand_raise_time/1000;
                ENDIF
                
                IF ok THEN
                    ok := Send("cmd accepted");
                    ! ok := Send(ValToStr(n_last_cmd) + " " + ValToStr(n_cur_cmd));
                ELSE
                    ok := Send("wrong cmd format");
                ENDIF
                
            ELSEIF cmd="EXIT" OR cmd="MoveL" OR cmd="MoveJ" OR cmd="MoveC" OR
                   cmd="Catch" OR cmd="SetRobotStartTime" THEN
                WaitUntil NOT(((s.n_last_cmd+1) MOD n_max_cmd) = (s.n_cur_cmd MOD n_max_cmd));
                
                new_cmd.type := cmd;
                IF cmd="MoveL" OR cmd="MoveJ" OR cmd="MoveC" THEN
                    pos11 := StrFind(recv_cmd, pos02+1, l);
                    pos12 := StrFind(recv_cmd, pos11+1, r);
                    tmp_str := StrPart(recv_cmd, pos11, pos12-pos11+1);
                    ok := StrToVal(tmp_str, new_cmd.target_robt1.trans);
                    IF ok AND new_cmd.type="MoveC" THEN
                        pos21 := StrFind(recv_cmd, pos12+1, l);
                        pos22 := StrFind(recv_cmd, pos21+1, r);
                        tmp_str := StrPart(recv_cmd, pos21, pos22-pos21+1);
                        ok := StrToVal(tmp_str, new_cmd.target_robt2.trans);
                    ENDIF
                ELSEIF cmd="Catch" THEN
                    
                    ! Add object_counter, 1;
                    ! ct := GetCurTime();
                    ! TPWrite "Object #" + ValToStr(object_counter) + ", ABB CurTime: " + ValToStr(ct); 
                    
                    pos11 := StrFind(recv_cmd, pos02+1, l);
                    pos12 := StrFind(recv_cmd, pos11+1, r);
                    pos12 := StrFind(recv_cmd, pos12+1, r);
                    tmp_str := StrPart(recv_cmd, pos11, pos12-pos11+1);
                    ok := StrToVal(tmp_str, tmp_catch);
                    new_cmd.target_robt1.trans := tmp_catch.trans;
                    new_cmd.target_robt2.trans := drop_pos{tmp_catch.item_type}; ! choose basket
                    new_cmd.time := tmp_catch.time/1000;
                ELSEIF cmd="SetRobotStartTime" THEN
                    pos11 := StrFind(recv_cmd, pos02+1, l);
                    pos12 := StrFind(recv_cmd, pos11+1, r);
                    tmp_str := StrPart(recv_cmd, pos11+1, pos12-pos11-1);
                    ok := StrToVal(tmp_str, new_cmd.time);
                    new_cmd.time := new_cmd.time/1000;
                    
                    RestartTimer;
                    start_time := new_cmd.time;
                ENDIF
                
                IF ok THEN
                    ok := AddNewCmd();
                    ok := Send("cmd accepted");
                ELSE
                    ok := Send("wrong cmd format");
                ENDIF
                
                IF cmd="EXIT" THEN
                    GOTO ExitTCPCmdLoopPart;
                ENDIF
            ELSE
                ok := Send("unknown cmd");
            ENDIF
        ENDWHILE
        ExitTCPCmdLoopPart:
        s.CmdLoopRunning := FALSE;
        TPWrite "Cmd Stop Time = ", \Num:=GetCurTime();
        StopTimer;
        
        ERROR
        IF ERRNO=ERR_SOCK_CLOSED THEN
            stop;
        ENDIF
        
        SocketClose clientSocket;
        SocketClose serverSocket;
        stop;
        
    ENDPROC
	PROC main()

	ENDPROC
	PROC RestartTimer()

	ENDPROC
	PROC StopTimer()

	ENDPROC
	PROC main()

	ENDPROC
	PROC RestartTimer()

	ENDPROC
	PROC StopTimer()

	ENDPROC
    
ENDMODULE
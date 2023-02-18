MODULE robot_arm
    
    PERS cmddata cmd_queue{n_max_cmd};
    PERS bool lock{n_max_cmd};
    VAR cmddata cur_cmd;
    PERS status s;
    VAR clock clk1;
    VAR num start_time;
    VAR num time_diff;
    
    CONST robtarget ROBT_DEFAULT := [[0,0,0],[1,0,0,0],[0,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    FUNC cmddata GetNewCmd()
        VAR num n_new_cmd;
        
        WaitUntil NOT (s.n_cur_cmd  - s.n_last_cmd = 0);
        
        n_new_cmd := s.n_cur_cmd+1;
        IF n_new_cmd=n_max_cmd+1 THEN
            n_new_cmd := 1;
        ENDIF
        
        WaitUntil lock{n_new_cmd}=FALSE;
        lock{n_new_cmd} := TRUE;
        cur_cmd := cmd_queue{n_new_cmd};
        lock{n_new_cmd} := FALSE;
        s.n_cur_cmd := n_new_cmd;
        
        RETURN cur_cmd;
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
        VAR num ct;
        VAR robtarget target;
        VAR num object_counter := 0;
        VAR robtarget tmp_rt;
        tmp_rt := Offs(ROBT_DEFAULT, 100, 0, 0);
        time_diff := 0.2;
        cur_cmd := ["None", 0, DUMMY_ROBT, DUMMY_ROBT, SD_DEFAULT, ZD_DEFAULT, DUMMY_HP, 0];
        FOR i FROM 1 TO n_max_cmd DO
            cmd_queue{i} := cur_cmd;
            lock{i} := FALSE;
        ENDFOR
        
        MoveJ Offs(ROBT_DEFAULT, 100, 0, 0),SD_DEFAULT,ZD_DEFAULT,MyTool\WObj:=wobj_ct;
        
        WaitUntil NOT (s.n_cur_cmd  - s.n_last_cmd = 0);
        
        RestartTimer;
        TPWrite "Robot Start Time = ", \Num:=GetCurTime();
        
        RACmdLoopPart:
        s.RobotLoopRunning := TRUE;
        WHILE TRUE DO                
            cur_cmd := GetNewCmd();
            !TPWrite "Cur:" + ValToStr(ct) + ",Catch:" + ValToStr(cur_cmd.time) + ",Wait: " + ValToStr(cur_cmd.time-ct);
            IF cur_cmd.type="Catch" THEN
                Add object_counter, 1;
                TPWrite "Catch Object #" + ValToStr(object_counter) + ",ABB CurTime: " + ValToStr(GetCurTime());
                
                ! Move to position to catch (h = hand_height)
                target := cur_cmd.target_robt1;
                target.trans.z := cur_cmd.hp.hand_height;
                MoveL target,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj_ct;

                ! Wait to start lower
                ct := GetCurTime();
                IF (ct < cur_cmd.time-cur_cmd.hp.hand_lower_time) THEN
                    WaitTime(cur_cmd.time-cur_cmd.hp.hand_lower_time-ct);
                ENDIF
                
                ! Lower (h = items_height)
                target.trans.z := cur_cmd.hp.items_height;
                MoveL target,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj_ct;
                
                ! Wait to catch
                ct := GetCurTime();
                IF (ct < cur_cmd.time) THEN
                    WaitTime(cur_cmd.time-ct);
                ENDIF
                
                ! Catch
                TPWrite "CATCH:", \Num:=GetCurTime();
                
                ! Raise (h = hand_height)
                target.trans.z := cur_cmd.hp.hand_height;
                MoveL target,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj_ct;
                
                ! Move to position to drop (h = hand_height)
                target := cur_cmd.target_robt2;
                target.trans.z := cur_cmd.hp.hand_height;
                MoveL target,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj_ct;
                
                ! Drop
                TPWrite "DROP:", \Num:=GetCurTime();
                
            ELSEIF cur_cmd.type="SetRobotStartTime" THEN
                RestartTimer;
                start_time := cur_cmd.time + time_diff;
                TPWrite "SetRobotStartTime Time = ", \Num:=start_time;
                
            ELSEIF cur_cmd.type="EXIT" THEN
                MoveJ ROBT_DEFAULT,SD_DEFAULT,ZD_DEFAULT,MyTool\WObj:=wobj_ct;
                GOTO ExitRACmdLoopPart;
            ELSEIF cur_cmd.type="MoveJ" THEN
                IF cur_cmd.target_wobj_num=1 THEN
                    MoveJ cur_cmd.target_robt1,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj_ct;
                ELSE
                    MoveJ cur_cmd.target_robt1,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj0;
                ENDIF
            ELSEIF cur_cmd.type="MoveL" THEN
                IF cur_cmd.target_wobj_num=1 THEN
                    MoveL cur_cmd.target_robt1,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj_ct;
                ELSE
                    MoveL cur_cmd.target_robt1,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj0;
                ENDIF
            ELSEIF cur_cmd.type="MoveC" THEN
                IF cur_cmd.target_wobj_num=1 THEN
                    MoveC cur_cmd.target_robt1,cur_cmd.target_robt2,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj_ct;
                ELSE
                    MoveC cur_cmd.target_robt1,cur_cmd.target_robt2,cur_cmd.sd,cur_cmd.zd,MyTool\WObj:=wobj0;
                ENDIF
            ENDIF
        ENDWHILE
        ExitRACmdLoopPart:
        s.RobotLoopRunning := FALSE;
        TPWrite "Robot Stop Time = ", \Num:=GetCurTime();
        StopTimer;
        
    ENDPROC
ENDMODULE
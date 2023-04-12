import socket
import time

def cmd_to_robot(sock, cmd):
    sock.send(cmd.encode('ASCII'))
    data = sock.recv(1024)
    print(data.decode('ASCII'))

def main():
    # -300 0 100
    #  300 0 100
    sock = socket.socket()
    sock.connect(("127.0.0.1", 1488))
    cmd_to_robot(sock, "MJ 0 0 100")

    cmd_to_robot(sock, "ROT -45")
    cmd_to_robot(sock, "MJ 0 0 20")
    cmd_to_robot(sock, "MJ 0 0 100")
    cmd_to_robot(sock, "ROT 45")
    time.sleep(0.5)
    cmd_to_robot(sock, "ROT 90")
    time.sleep(0.5)
    cmd_to_robot(sock, "ROT 90")
    time.sleep(0.5)
    cmd_to_robot(sock, "ROT 90")
    time.sleep(0.5)
    cmd_to_robot(sock, "ROT_BASE ")
    cmd_to_robot(sock, "MJ 0 0 100")

if __name__ == '__main__':
    main()

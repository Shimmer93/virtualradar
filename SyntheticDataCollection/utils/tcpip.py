import socket

class TCPServer:
    def __init__(self, host, port, timeout=120, verbose=False):
        self.host = host
        self.port = port
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.timeout = timeout
        self.verbose = verbose
        self.conn = None
        self.addr = None

        if self.verbose:
            print('Server is running on IP: {} and port: {}'.format(self.host, self.port))
            print('Server is waiting for connection...')

    def start_server(self):
        self.tcp_server.bind((self.host, self.port))
        self.tcp_server.settimeout(self.timeout)
        self.tcp_server.listen(1)
        self.conn, self.addr = self.tcp_server.accept()
        self.conn.settimeout(self.timeout)
        if self.verbose:
            print('Connected to: ', self.addr)
        
    def receive_data(self, datalengthbyte):
        data = self.conn.recv(datalengthbyte)
        if self.verbose:
            print('Received data with length: ', len(data))
        return data
    
    def receive_data_flexible(self):
        datalengthbyte = int.from_bytes(self.conn.recv(4), byteorder='little')
        data = self.conn.recv(datalengthbyte)
        if self.verbose:
            print('Received data with length: ', len(data))
        return data
    
    def close_server(self):
        self.conn.close()
        self.tcp_server.close()
        if self.verbose:
            print('Server is closed')
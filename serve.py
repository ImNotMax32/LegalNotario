from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    PORT = 8000
    ip = get_ip()
    
    print(f"\nDémarrage du serveur sur : http://{ip}:{PORT}")
    print("Pour voir les données, ouvrez : ")
    print(f"http://{ip}:{PORT}/view_data.html")
    print("\nCtrl+C pour arrêter le serveur")
    
    httpd = HTTPServer(('0.0.0.0', PORT), CORSRequestHandler)
    httpd.serve_forever()

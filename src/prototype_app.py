"""Serve the local app at http://127.0.0.1:8765."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse
import json
from prototype_engine import PrototypeEngine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    engine=PrototypeEngine()
    class Handler(BaseHTTPRequestHandler):
        def reply(self,data,status=200,content_type='application/json; charset=utf-8'):
            body=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path=='/':
                self.reply(Path(__file__).with_name('prototype_ui.html').read_bytes(),content_type='text/html; charset=utf-8')
            elif self.path=='/api/bootstrap':
                self.reply(engine.bootstrap())
            elif self.path=='/api/health':
                self.reply({'status':'ok','baseline':'Run14 v1.1'})
            else:
                self.reply({'error':'Not found'},404)
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536:
                    raise ValueError('Invalid request size.')
                data=json.loads(self.rfile.read(size))
                if self.path=='/api/extract':
                    self.reply(engine.extract(data.get('query',''),allow_live=data.get('allow_live') is True))
                elif self.path=='/api/recommend':
                    self.reply(engine.recommend(data['user_id'],data['intent'],data.get('category') or None,int(data.get('top_k',5))))
                else:
                    self.reply({'error':'Not found'},404)
            except (ValueError,KeyError,TypeError) as exc:
                self.reply({'error':str(exc)},400)
            except Exception as exc:
                self.reply({'error':'Request incomplete: '+str(exc)},503)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Prototype running: http://127.0.0.1:{args.port} (Ctrl+C to stop)',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()

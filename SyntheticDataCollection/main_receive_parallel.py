import argparse
from main_receive import main
import threading

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Receive radar data from multiple transmitters')
    parser.add_argument('-a', '--host', type=str, default='localhost', help='Host IP address')
    parser.add_argument('-p', '--ports', type=int, nargs='+', default=[55000, 55001, 55002], help='Port numbers')
    parser.add_argument('-c', '--cfg_path', type=str, default='D:\\Repositories\\virtualradar\\SyntheticDataCollection\\cfg\\ti_xwr1843.yml', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print debug information')
    args = parser.parse_args()

    ts = []
    for port in args.ports:
        args_i = argparse.Namespace(host=args.host, port=port, cfg_path=args.cfg_path, verbose=args.verbose)
        ti = threading.Thread(target=main, args=(args_i,))
        ts.append(ti)
    [t.start() for t in ts]
    [t.join() for t in ts]
    print('All threads finished')

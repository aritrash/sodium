from __future__ import annotations
import argparse
from pathlib import Path
from .pipeline import inventory, run_profile, build_pchip, review_curve, clean_curve

def main():
    parser=argparse.ArgumentParser(prog="sodium-digitizer",description="Sodium tube-characteristic digitization and dataset pipeline")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("inventory",help="Inventory source images"); p.add_argument("--input",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p=sub.add_parser("classify",help="Classify images from filename metadata"); p.add_argument("--input",required=True,type=Path); p.add_argument("--inventory",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p=sub.add_parser("run",help="Run LinePilot using a graph profile"); p.add_argument("--input",required=True,type=Path); p.add_argument("--profile",required=True,type=Path); p.add_argument("--linepilot",required=True,type=Path); p.add_argument("--linepilot-python",type=Path); p.add_argument("--output",required=True,type=Path)
    p=sub.add_parser("validate",help="Create a review candidate; --accept explicitly records manual validation"); p.add_argument("--input",required=True,type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--x-column",default="x"); p.add_argument("--y-column",default="y"); p.add_argument("--accept",action="store_true",help="Explicitly assert that the curve has been manually reviewed"); p.add_argument("--reviewer",default="user")
    p=sub.add_parser("clean",help="Manually exclude selected points or X intervals; writes an audit log"); p.add_argument("--input",required=True,type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--x-column",default="x"); p.add_argument("--y-column",default="y"); p.add_argument("--drop-indexes",help="Zero-based row indices/ranges, e.g. 4,8-11"); p.add_argument("--drop-x-ranges",help="Inclusive X intervals, e.g. 180:200,250:270")
    p=sub.add_parser("pchip",help="PCHIP-resample manually validated candidate curves only"); p.add_argument("--input",required=True,type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--step",type=float,default=1.0); p.add_argument("--x-column",default="x"); p.add_argument("--y-column",default="y")
    args=parser.parse_args()
    if args.command=="inventory": inventory(args.input,args.output)
    elif args.command=="classify":
        from .pipeline import classify_inventory
        classify_inventory(args.input,args.inventory,args.output)
    elif args.command=="run": run_profile(args.input,args.profile,args.linepilot,args.output,args.linepilot_python)
    elif args.command=="validate": review_curve(args.input,args.output,args.x_column,args.y_column,args.accept,args.reviewer)
    elif args.command=="clean": clean_curve(args.input,args.output,args.x_column,args.y_column,args.drop_indexes,args.drop_x_ranges)
    elif args.command=="pchip": build_pchip(args.input,args.output,args.step,args.x_column,args.y_column)

if __name__=="__main__": main()

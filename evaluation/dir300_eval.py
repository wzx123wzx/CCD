import argparse
import json
import multiprocessing as mp
import os


def get_version():
    """
    Returns the version of the various packages used for evaluation.
    """
    import pytesseract

    return {
        "tesseract": str(pytesseract.get_tesseract_version()),
        "pyesseract": os.popen("pip list | grep pytesseract").read().split()[-1],
        "Levenshtein": os.popen("pip list | grep Levenshtein").read().split()[-1],
        "jiwer": os.popen("pip list | grep jiwer").read().split()[-1],
        "matlabengineforpython": os.popen("pip list | grep matlab").read().split()[-1],
    }


def visual_metrics_process(queue, dir300_path, preds_path, verbose):
    """
    Subprocess function that computes visual metrics (MS-SSIM, LD, and AD) based on a matlab script.
    """
    import matlab.engine

    eng = matlab.engine.start_matlab()
    eng.cd(r"./eval/eval_code/", nargout=0)
    mean_ms, mean_ld, mean_ad = eng.evalScriptDIR300(dir300_path, preds_path, verbose, nargout=3)
    queue.put(dict(ms=mean_ms, ld=mean_ld, ad=mean_ad))


def ocr_process(queue, dir300_path, preds_path, inputs_path):
    """
    Subprocess function that computes OCR metrics (CER and ED).
    """
    from eval.ocr_eval.ocr_eval import OCR_eval_DIR300

    CERmean, EDmean, OCR_dict_results = OCR_eval_DIR300(dir300_path, preds_path, inputs_path)
    with open(os.path.join(preds_path, "ocr_res.json"), "w") as f:
        json.dump(OCR_dict_results, f)
    queue.put(dict(cer=CERmean, ed=EDmean))


def compute_metrics(dir300_path, pred_path, inputs_path, save_path, verbose=False):
    """
    Compute and save all metrics.
    """
    if not pred_path.endswith("/"):
        pred_path += "/"
    q = mp.Queue()

    # Create process to compute MS-SSIM, LD, AD
    p1 = mp.Process(
        target=visual_metrics_process,
        args=(q, os.path.join(dir300_path), os.path.join(pred_path), verbose),
    )
    p1.start()

    # Create process to compute OCR metrics
    p2 = mp.Process(
        target=ocr_process, args=(q, os.path.join(dir300_path), os.path.join(pred_path), inputs_path)
    )
    p2.start()

    p1.join()
    p2.join()

    # Get results
    res = {}
    for _ in range(q.qsize()):
        ret = q.get()
        for k, v in ret.items():
            res[k] = v

    # Print and saves results
    print("--- Results ---")
    print(f"  Mean MS-SSIM      : {res['ms']}")
    print(f"  Mean LD           : {res['ld']}")
    print(f"  Mean AD           : {res['ad']}")
    print(f"  Mean CER          : {res['cer']}")
    print(f"  Mean ED           : {res['ed']}")

    with open(os.path.join(save_path, "dir300_res.txt"), "w") as f:
        f.write(f"Mean MS-SSIM      : {res['ms']}\n")
        f.write(f"Mean LD           : {res['ld']}\n")
        f.write(f"Mean AD           : {res['ad']}\n")
        f.write(f"Mean CER          : {res['cer']}\n")
        f.write(f"Mean ED           : {res['ed']}\n")

        # f.write("\n--- Module Version ---\n")
        # for module, version in get_version().items():
        #     f.write(f"{module:25s}: {version}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dewarping results on DIR300 benchmark.")

    parser.add_argument(
        "--dir300-path", type=str,
        default="/root/autodl-tmp/dataset/DIR300/gt",
        help="Path to the DIR300 GT directory (e.g. /root/autodl-tmp/dataset/DIR300/gt)"
    )
    parser.add_argument(
        "--pred-path", type=str,
        default="sampling/dir300/test/pred_dewarped",
        help="Path to the directory containing dewarped prediction images (e.g. sampling/dir300/test/pred_dewarped)"
    )
    parser.add_argument(
        "--inputs-path", type=str,
        default="/root/autodl-tmp/dataset/DIR300/dist",
        help="Path to the DIR300 input (distorted) images directory (e.g. /root/autodl-tmp/dataset/DIR300/dist)"
    )
    parser.add_argument(
        "--save-path", type=str,
        default="sampling/dir300/test",
        help="Directory to save the result file dir300_res.txt (e.g. sampling/dir300/test)"
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print verbose output during evaluation")

    args = parser.parse_args()
    compute_metrics(
        dir300_path=os.path.abspath(args.dir300_path),
        pred_path=os.path.abspath(args.pred_path),
        inputs_path=os.path.abspath(args.inputs_path),
        save_path=os.path.abspath(args.save_path),
        verbose=args.verbose,
    )

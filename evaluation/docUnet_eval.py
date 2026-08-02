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

def visual_metrics_process(queue, docunet_path, preds_path, verbose):
    """
    Subprocess function that computes visual metrics (MS-SSIM, LD, and AD) based on a matlab script.
    """
    import matlab.engine

    eng = matlab.engine.start_matlab()
    eng.cd(r"./eval/eval_code/", nargout=0)

    mean_ms, mean_ld, mean_ad = eng.evalScriptDocUNet(docunet_path, preds_path, verbose, nargout=3)
    queue.put(dict(ms=mean_ms, ld=mean_ld, ad=mean_ad))


def ocr_process(queue, docunet_path, preds_path, inputs_path):
    """
    Subprocess function that computes OCR metrics (CER and ED).
    """
    from eval.ocr_eval.ocr_eval import OCR_eval_docunet

    CERmean, EDmean, OCR_dict_results = OCR_eval_docunet(
        docunet_path, preds_path, inputs_path)
    with open(os.path.join(preds_path, "ocr_res.json"), "w") as f:
        json.dump(OCR_dict_results, f)
    queue.put(dict(cer=CERmean, ed=EDmean))


def compute_metrics(docunet_path, preds_path, inputs_path, save_path, verbose=False):
    """
    Compute and save all metrics.
    """
    if not preds_path.endswith("/"):
        preds_path += "/"
    q = mp.Queue()

    # Create process to compute MS-SSIM, LD, AD
    p1 = mp.Process(target=visual_metrics_process, args=(q, docunet_path, preds_path, verbose))
    p1.start()

    # Create process to compute OCR metrics
    p2 = mp.Process(target=ocr_process, args=(q, docunet_path, preds_path, inputs_path))
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

    with open(os.path.join(save_path, "docunet_res.txt"), "w") as f:
        f.write(f"Mean MS-SSIM      : {res['ms']}\n")
        f.write(f"Mean LD           : {res['ld']}\n")
        f.write(f"Mean AD           : {res['ad']}\n")
        f.write(f"Mean CER          : {res['cer']}\n")
        f.write(f"Mean ED           : {res['ed']}\n")

        # model_info_path = os.path.join(preds_path, "model_info.txt")
        # if os.path.isfile(model_info_path):
        #     with open(model_info_path) as modinf_f:
        #         for x in modinf_f.readlines():
        #             f.write(x)

        # f.write("\n--- Module Version ---\n")
        # for module, version in get_version().items():
        #     f.write(f"{module:25s}: {version}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dewarping results on DocUNet benchmark.")

    parser.add_argument(
        "--docunet-path", type=str,
        default="/root/autodl-tmp/dataset/DocUNet/scan",
        help="Path to the DocUNet scan (GT) directory (e.g. /root/autodl-tmp/dataset/DocUNet/scan)"
    )
    parser.add_argument(
        "--pred-path", type=str,
        default="sampling/docunet/test/pred_dewarped",
        help="Path to the directory containing dewarped prediction images (e.g. sampling/docunet/test/pred_dewarped)"
    )
    parser.add_argument(
        "--inputs-path", type=str,
        default="/root/autodl-tmp/dataset/DocUNet/crop",
        help="Path to the DocUNet input (cropped) images directory (e.g. /root/autodl-tmp/dataset/DocUNet/crop)"
    )
    parser.add_argument(
        "--save-path", type=str,
        default="sampling/docunet/test",
        help="Directory to save the result file docunet_res.txt (e.g. sampling/docunet/test)"
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print verbose output during evaluation")

    args = parser.parse_args()
    compute_metrics(
        docunet_path=os.path.abspath(args.docunet_path),
        preds_path=os.path.abspath(args.pred_path),
        inputs_path=os.path.abspath(args.inputs_path),
        save_path=os.path.abspath(args.save_path),
        verbose=args.verbose,
    )

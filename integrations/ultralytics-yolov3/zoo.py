from sparsezoo import main as sparsezoo_main


def main():
    sparsezoo_main(
        domain="cv",
        sub_domain="detection",
        architecture="yolo_v3",
        framework="pytorch",
        repo="ultralytics",
    )


if __name__ == "__main__":
    main()

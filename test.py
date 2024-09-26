from PromptIQA import run_promptIQA

if __name__ == "__main__":
    promptIQA = run_promptIQA.PromptIQA()

    ISPP_I = [
        "./Examples/Example1/ISPP/1600.AWGN.1.png",
        "./Examples/Example1/ISPP/1600.AWGN.2.png",
        "./Examples/Example1/ISPP/1600.AWGN.3.png",
        "./Examples/Example1/ISPP/1600.AWGN.4.png",
        "./Examples/Example1/ISPP/1600.AWGN.5.png",
        "./Examples/Example1/ISPP/1600.BLUR.1.png",
        "./Examples/Example1/ISPP/1600.BLUR.2.png",
        "./Examples/Example1/ISPP/1600.BLUR.3.png",
        "./Examples/Example1/ISPP/1600.BLUR.4.png",
        "./Examples/Example1/ISPP/1600.BLUR.5.png",
    ]

    ISPP_S = [0.062, 0.206, 0.262, 0.375, 0.467, 0.043, 0.142, 0.341, 0.471, 0.75]
    Image = "./Examples/Example1/cactus.png"

    score = promptIQA.run(ISPP_I, ISPP_S, Image)
    
    print(score)

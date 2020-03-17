import options
import train, predict



if __name__ == "__main__":
    args = options.parseArguments()
    
    if args.mode=="train":
        train.train(args)

    elif args.mode=="predict":
        predict.predict(args)

    else:
        raise NotImplementedError
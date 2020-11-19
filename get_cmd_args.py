import argparse



def get_cmd_args():
    parser = argparse.ArgumentParser()
     # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    #parser.add_argument("--train_dir", default="flowers/data_dir/train" ,help="data directory , default train directory")
    
    parser.add_argument("data_dir" , default="flowers", help="image directory for training - default flowers ")
    parser.add_argument("--arch" , default="vgg16", help="Model to use default is vgg16, Can use ONLY vgg16 OR vgg13 or alexnet ")
    parser.add_argument("--gpu", action="store_true",default=False,help="optional default is False")
    parser.add_argument("--save_dir" , default="save_dir", help="directory to save checkpoint.pth. ")
    
    parser.add_argument("--learning_rate",default=".001" ,type=float,help="Learning rate to use")
    parser.add_argument("--epochs",default = "1" ,type=int,help="number of epochs to use , default is'1'")
    parser.add_argument("--batch_size",default="32",type=int, help = "batch size to use default is '32'")
    parser.add_argument("--hidden_units",default="512",type=int, help = "hidden_units size to use default is '512'")
    args = parser.parse_args()
    
      
       
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return args
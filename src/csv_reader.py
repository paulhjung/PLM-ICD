import csv 
def read_csv_line_by_line(file_path): 
    with open(file_path, 'r') as file: 
        reader = csv.reader(file) 
        for row in reader: 
            yield row 
            
def main(): 
    file_path = input("Enter the path to your CSV file: ") 
    line_generator = read_csv_line_by_line(file_path) 
    while True: 
        input("Press Enter to get the next line...") 
        try: 
            print(next(line_generator)) 
        except StopIteration: 
            print("End of file reached.") 
            break 
if __name__ == "__main__": main()
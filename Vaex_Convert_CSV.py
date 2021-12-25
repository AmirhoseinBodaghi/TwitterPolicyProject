import vaex
address_input = '/mnt/tb/amirdata/Merge_All_Waves.zip'
#-------------
def main():
    vaex_df = vaex.from_csv(address_input, convert = True, compression='zip', encoding="ISO-8859-1", engine='python')
#-----------
if __name__ == '__main__':
        main ()

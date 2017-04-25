import glob
import pickle

vehiclesPath = glob.glob('../data/vehicles/**/*.png')
nonVehPath   = glob.glob('../data/non-vehicles/**/*.png')

print (vehiclesPath)
pickle_file = '../data/allData.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'vehicles': vehiclesPath,
                'nonVeh': nonVehPath    
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
    
print('Data cached in pickle file.')
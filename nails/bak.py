## backed up from model_rec.py
for f in flist:
    hm = Hand_Model(f)
    nail_size_vec = []
    for i in range(5):
        vf = hm.full_cross_model[i]
        min_dist = 9999
        min_idx = 0
        for j in range( NUM_COMBI ):
            dist = cm_set.combi_models[j].geometric_distance_to( vf, i )
            # d1 = distance.euclidean( vh, vm1)
            print( 'Finger {} Model {} - Euclidean distance: {:07.2f} full cross'.format( i, j, dist) )
            if (  dist <= min_dist):
                min_dist = dist
                min_idx = j
        if i == 4 :
            nail_size_vec.append(SIZE_CHART[i][min_idx])
            print("Thumb is classified to combi id {} in {} - size {} ".format( min_idx,
                                                        COMBI_THUMBS[min_idx], SIZE_CHART[4][min_idx]))
        else:
            sz = SIZE_CHART[i][min_idx]
            nail_size_vec.append(sz)
            print( "Finger {} is classified to combi id {} in {} - size {}".format( i,
                                                        min_idx, COMBI_FINGERS[min_idx], sz ))
    nschart = np.array(SIZE_CHART)
    min_dist = 9999
    min_idx = 0
    dist_vec = []
    for i in range(NUM_COMBI):
        combi_size_vec = nschart[:,i]
        dist = distance.euclidean( nail_size_vec, combi_size_vec )
        dist_vec.append(dist)
        if dist < min_dist :
            min_dist = dist
            min_idx = i
    print( "Nail size classification assigns the hand to combi {}".format( min_idx + 1))
    print( "Nails classified to sizes: ", nail_size_vec )
    print()
# Head-Movement-Prediction-of-Audience-for-360-Panoramic-Video

[video.py](https://github.com/Yiran-H/Head-Movement-Prediction-of-Audience-for-360-Panoramic-Video/blob/main/video.py): global variable that reads the prediction settings file from config.ini and the video information file from dash360.ini, providing video properties and parameters for all program files

[transformtile.py](https://github.com/Yiran-H/Head-Movement-Prediction-of-Audience-for-360-Panoramic-Video/blob/main/transformtile.py): input yaw, pitch values, give all the tile IDs within the field of view

[prediction.py](https://github.com/Yiran-H/Head-Movement-Prediction-of-Audience-for-360-Panoramic-Video/blob/main/prediction.py): is the underlying file for predictpolicy. Given an array of horizontal and vertical coordinates and the horizontal coordinates to be predicted, the prediction can be made.

[predictpolicy.py](https://github.com/Yiran-H/Head-Movement-Prediction-of-Audience-for-360-Panoramic-Video/blob/main/predictpolicy.py): mainly does data splitting and underlying policy selection, input is a three-column matrix, split into two two-column matrices [frame_no, yaw] [frame_no, pitch]. Of course, in addition to this, some underlying strategy selection can be done, for example, multiple underlying strategies can be called at the same time.

[accuracy.py](https://github.com/Yiran-H/Head-Movement-Prediction-of-Audience-for-360-Panoramic-Video/blob/main/accuracy.py): this is mainly responsible for accuracy calculation, there is general accuracy, that is, set the tolerance threshold, exceed the prediction error; there is also tileaccuracy, that is, according to the calculation of the correct tile to determine the accuracy is how much.

[database_player.py](https://github.com/Yiran-H/Head-Movement-Prediction-of-Audience-for-360-Panoramic-Video/blob/main/database_player.py): this is very important, the input is the file name and the policy (predictpolicy.py).
For example, HMDplayer(file_in, pp.WLRpolicy) can simulate the playback data, using a weighted linear regression.
In addition, HMDplayer can also provide some necessary data, such as an array of predictions and an array of actual values, and can also call accuracy to calculate the accuracy.
In addition, the draw method inside can be used for plotting

[policy_compare.py](https://github.com/Yiran-H/Head-Movement-Prediction-of-Audience-for-360-Panoramic-Video/blob/main/policy_compare.py): usually used to compare the difference between different policies for the same user
Please change the user_no, which refers to the first user, for example, if user_no = 1 then the data of user 1 is used
Policy list policy = [Policy(LSRpolicy), Policy(LRpolicy), Policy(WLRpolicy)], here you can add the policy you use here

import cv2
import numpy as np
import time
#from operator import itemgetter
import os
import Cards
import PiVideoStream
from picamera.array import PiRGBArray 
from picamera import PiCamera

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

frame_rate_calc = 1
freq = cv2.getTickFrequency()

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera.
videostream = PiVideoStream.PiVideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE).start()
time.sleep(1) # warm up

# Load the rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')

cam_quit = 0 # Loop control variable

while cam_quit == 0:
    image = videostream.read()
    t1 = cv2.getTickCount()
    # Pre-process
    pre_proc = Cards.preprocess_image(image)	
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)
    if len(cnts_sort) != 0:
        cards = []
        k = 0
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                cards.append(Cards.preprocess_card(cnts_sort[i],image))
                cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)
                image = Cards.draw_results(image, cards[k])
                k = k + 1
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
            if (len(cards) == 5):
                poker_cards = []
                is_playing_card = 0
                for i in range(len(cards)):
                    if((cards[i].best_rank_match != "Unknown") and (cards[i].best_suit_match != "Unknown")):
                        poker_cards.append(cards[i])
                        is_playing_card = is_playing_card + 1
                if(is_playing_card == 5):
                    #print("Poker Confirmed")
                    trans_card = Cards.transition(poker_cards)
                    #print(trans_card[4][1])
                    # sort by rank
                    sort_card = sorted(trans_card, key=lambda card: card[0])
                    #print(sort_card[1][0])
                    #print(sort_card[1][1])
                    count_suit_poker = Cards.count_suit(sort_card)
                    count_card_poker = Cards.count_card(sort_card)
                    #for i in range(len(count_card_poker)):
                    #    print(count_card_poker[i][0])
                    #    print(count_card_poker[i][1])
                    #for i in range(len(count_suit_poker)):
                    #    print(count_suit_poker[i][0])
                    #    print(count_suit_poker[i][1])
                    #print(len(count_suit_poker))
                    #print(len(count_card_poker))
                    result = Cards.poker_result(count_card_poker,count_suit_poker)
                    cv2.putText(image,result,(525,26),font,0.7,(0,200,0),2,cv2.LINE_AA)
                    #print(result) 
                    cv2.putText(image,"Poker Confirmed",(1050,26),font,0.7,(100,200,200),2,cv2.LINE_AA)   
                #else:
                #   print("No Poker")
        
    cv2.putText(image,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

    cv2.imshow("Card Detector",image)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    
    # If 'q' is pressed then exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
        
cv2.destroyAllWindows()
videostream.stop()


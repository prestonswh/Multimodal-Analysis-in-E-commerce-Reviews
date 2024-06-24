* Load the CSV file

* /Users/ojeongsig/Desktop/yeonguno/did_test/df_review_resnet_merge_processed.csv : pretrained resnet, not finetuning 




* /Users/ojeongsig/Desktop/yeonguno/did_test/resnet_helpful_processed.csv: helpful resnet152 sentence transformer -> 이미지 짱 resnet152 + text짱 sentence transformer(all-MiniLM-L6-v2)  
* /Users/ojeongsig/Desktop/yeonguno/resnet_helpful_sentiment_processed.csv -> sentiment(distilbert-base-uncased-finetuned-sst-2-english) resenet transformer, 

* difference t- (t-1) delta를 본다. 


import delimited "my_data_files/resnet_helpful_processed.csv", clear


drop if helpful_vote <= 1

gen real_datetime = clock(datetime, "DMYhms")
format real_datetime %tc

encode product_id, gen(product_id_num)

* Check for duplicates again with more detailed listing
bysort product_id_num real_datetime: gen dup_id = _n
list product_id_num real_datetime dup_id if dup_id > 1, noobs

* Drop duplicates where dup_id is 2
drop if dup_id == 2

* Drop the dup_id variable as it is no longer needed
drop dup_id

* Handle potential outliers
* For example, winsorize rating at 1% and 99%
gen rating_winsor = helpful_vote
egen rating_low = pctile(helpful_vote), p(1)
egen rating_high = pctile(helpful_vote), p(99)
replace rating_winsor = rating_low if helpful_vote < rating_low
replace rating_winsor = rating_high if helpful_vote > rating_high

xtset product_id_num real_datetime

gen remainder = mod(month, 12)

* \
* Regression without interaction term
* xtreg rating_winsor volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)

* Regression with treatment interaction (AfterTreat)
* xtreg rating_winsor aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)

* Adding interaction between similarity and AfterTreat to the model
gen similarity_x_aftertreat = pair_similarity * aftertreat
xtreg rating_winsor aftertreat similarity_x_aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.remainder, fe vce(cluster product_id_num)


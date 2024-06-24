* Load the CSV file


* /Users/ojeongsig/Desktop/yeonguno/resnet_helpful_sentiment_processed.csv -> sentiment analysis (distilbert-base-uncased-finetuned-sst-2-english) resenet transformer -> 0.38 -> 2개의 lable에 대해서만 분류 

* /Users/ojeongsig/Desktop/yeonguno/did_test/resnet_helpful_sentiment_processed2.csv -> bert-base-multilingual-uncased-sentiment analysis -> lable이 star1,.. ,star5로 5개로 구성
* difference t- (t-1) delta를 본다. 



* Load the CSV file
import delimited "/Users/ojeongsig/Desktop/yeonguno/did_test/resnet_helpful_sentiment_processed2.csv", clear

* Check if the data is loaded correctly
list in 1/10

* Convert datetime
gen real_datetime = clock(datetime, "DMYhms")
format real_datetime %tc

* Encode product_id
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
gen rating_winsor = sentiment_score

* Convert sentiment labels to numeric values for regression analysis


* Set panel data structure
xtset product_id_num real_datetime

gen remainder = mod(month, 12)

* Check the data structure and contents
describe
summarize

* Ensure there are observations left
count

* If there are observations, proceed with regression
if r(N) > 0 {
    * Regression without interaction term
    xtreg rating_winsor volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.remainder, fe vce(cluster product_id_num)

    * Regression with treatment interaction (AfterTreat)
    xtreg rating_winsor aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.remainder, fe vce(cluster product_id_num)

    * Adding interaction between similarity and AfterTreat to the model
    gen similarity_x_aftertreat = pair_similarity * aftertreat
    xtreg rating_winsor aftertreat similarity_x_aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.remainder, fe vce(cluster product_id_num)
} else {
    di "No observations left after data processing steps."
}

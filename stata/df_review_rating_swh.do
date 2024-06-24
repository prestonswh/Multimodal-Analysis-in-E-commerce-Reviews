* Load the CSV file

* /Users/swh/Desktop/kaistms/24_1/연구고급논제_BME785/Proposal/my_data_files/df_review_resnet_merge_processed.csv : pretrained resnet50, not finetuning -> image similarity만 존재 -> 성능이 꽤 괜찮았음

* /Users/swh/Desktop/kaistms/24_1/연구고급논제_BME785/Proposal/my_data_files/merged_file_processed.csv: image: not resnet -> fashion dataset에 대해서 pretrained 사용 -> 오히려 p value 안좋음

* /Users/swh/Desktop/kaistms/24_1/연해고곱논제_BME785/Proposal/my_data_files/df_review_resnet_merge_resnet2_processed.csv : resnet + stsb-roberta-large' -> text bad

* /Users/swh/Desktop/kaistms/24_1/연해고곱논제_BME785/Proposal/my_data_files/df_review_resnet152_merge_resnet2_processed.csv: resnet152 + sentence transformer 0.02 -> sentence transformer 못찾음 ㅠㅠ

* /Users/swh/Desktop/kaistms/24_1/연해고곱논제_BME785/Proposal/my_data_files/df_review_resnet152_merge_resnet_y_processed.csv: RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1 + sentence transformer -> resnet_y\\\ -> resnet152보다 더 복잡한 모델이지만 실패

* /Users/swh/Desktop/kaistms/24_1/연해고곱논제_BME785/Proposal/my_data_files/df_review_resnet152_merge_resnet_vit_processed.csv visual transformer + sentence transformer(all-MiniLM-L6-v2) -> 
* -> visual transformer 성능이 아쉬웠다.
* /Users/swh/Desktop/kaistms/24_1/연해고곱논제_BME785/Proposal/my_data_files/df_review_resnet152_merge_resnet_efficient_processed.csv: efficientnet + sentence transformer(all-MiniLM-L6-v2) -> 
* efficientnet 되게 무거운 모델을 사용했지만 성능은 아쉬웠다.

* /Users/swh/Desktop/kaistms/24_1/연해고곱논제_BME785/Proposal/my_data_files/df_review_resnet152_merge_resnet_mpnet_processed.csv : resnet152 + mpnet (sentence transformer 성능아쉬움) -> sentence-transformers/all-mpnet-base-v2

* my_data_files/resnet_50_06_14_processed.csv

* difference t- (t-1) delta를 본다. 
import delimited "my_data_files/resnet_50_06_14_processed.csv", clear

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
gen rating_winsor = rating
egen rating_low = pctile(rating), p(1)
egen rating_high = pctile(rating), p(99)
replace rating_winsor = rating_low if rating < rating_low
replace rating_winsor = rating_high if rating > rating_high

xtset product_id_num real_datetime

gen remainder = mod(month, 12)

* \
* Regression without interaction term
* xtreg rating_winsor volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)

* Regression with treatment interaction (AfterTreat)
* xtreg rating_winsor aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)

* Adding interaction between similarity and AfterTreat to the model
gen similarity_x_aftertreat = image_similarity * aftertreat
xtreg rating aftertreat similarity_x_aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.remainder, fe vce(cluster product_id_num)

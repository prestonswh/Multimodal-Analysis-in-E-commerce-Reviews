* If using directly after importing
import delimited "treatment_months.csv", clear
save treatment_months.dta, replace

* Load your main dataset if not already loaded
import delimited "df_review_processed.csv", clear

* Merge the treatment month into the main dataset
merge m:1 product_id using treatment_months.dta

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

* Setting up panel data structure
xtset product_id_num real_datetime

* Generate the Preit and Postit indicator variables
forvalues j = 1/5 {
    gen Preit`j' = (month == treatment_month - `j')
    gen Postit`j' = (month == treatment_month + `j')
}

* Generate cumulative indicators for >=6 months before and after treatment
gen Preit6 = (month <= treatment_month - 6)
gen Postit6 = (month >= treatment_month + 6)

* Run the fixed effects regression including the dummy variables
xtreg rating i.Preit2 i.Preit3 i.Preit4 i.Preit5 i.Preit6 i.Postit* volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)

xtreg volume aftertreat i.mon, vce(cluster product_id_num)




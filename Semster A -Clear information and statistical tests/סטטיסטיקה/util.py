from datetime import date

def date_diff(a , b, target):

    year,month,day = a.year,a.month,a.day
    d0 = date(year,month,day)

    year,month,day = b.year,b.month,b.day
    d1= date(year,month,day)

    delta = d1 - d0
    if abs(delta.days) == target:
        return True
    else : return False
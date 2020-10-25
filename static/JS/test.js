// Grab the datetime value from the filter
let school = d3.select("#school").property("value");
let schoolId = d3.select("#school").property("id");

let sex = d3.select("#sex").property("value");
let sexId = d3.select("#sex").property("id");

let age = d3.select("#age").property("value");
let ageId = d3.select("#age").property("id");

let address = d3.select("#address").property("value");
let addressId = d3.select("#address").property("id");

let famsize = d3.select("#famsize").property("value");
let famsizeId = d3.select("#famsize").property("id");

let Pstatus = d3.select("#Pstatus").property("value");
let PstatusId = d3.select("#Pstatus").property("id");

let Medu = d3.select("#Medu").property("value");
let MeduId = d3.select("#Medu").property("id");

let Fedu = d3.select("#Fedu").property("value");
let FeduId = d3.select("#Fedu").property("id");

let Mjob = d3.select("#Mjob").property("value");
let MjobId = d3.select("#Mjob").property("id");

let Fjob = d3.select("#Fjob").property("value");
let FjobId = d3.select("#Fjob").property("id");

let reason = d3.select("#reason").property("value");
let reasonId = d3.select("#reason").property("id");

let guardian = d3.select("#guardian").property("value");
let guardianId = d3.select("#guardian").property("id");

let traveltime = d3.select("#traveltime").property("value");
let traveltimeId = d3.select("#traveltime").property("id");

let studytime = d3.select("#studytime").property("value");
let studytimeId = d3.select("#studytime").property("id");

let failures = d3.select("#failures").property("value");
let failuresId = d3.select("#failures").property("id");

let schoolsup = d3.select("#schoolsup").property("value");
let schoolsupId = d3.select("#schoolsup").property("id");

let famsup = d3.select("#famsup").property("value");
let famsupId = d3.select("#famsup").property("id");

let paid = d3.select("#paid").property("value");
let paidId = d3.select("#paid").property("id");

let activities = d3.select("#activities").property("value");
let activitiesId = d3.select("#activities").property("id");

let nursery = d3.select("#nursery").property("value");
let nurseryId = d3.select("#nursery").property("id");

let higher = d3.select("#higher").property("value");
let higherId = d3.select("#higher").property("id");

let internet = d3.select("#internet").property("value");
let internetId = d3.select("#internet").property("id");

let romantic = d3.select("#romantic").property("value");
let romanticId = d3.select("#romantic").property("id");

let famrel = d3.select("#famrel").property("value");
let famrelId = d3.select("#famrel").property("id");

let freetime = d3.select("#freetime").property("value");
let freetimeId = d3.select("#freetime").property("id");

let goout = d3.select("#goout").property("value");
let gooutId = d3.select("#goout").property("id");

let Dalc = d3.select("#Dalc").property("value");
let DalcId = d3.select("#Dalc").property("id");

let Walc = d3.select("#Walc").property("value");
let WalcId = d3.select("#Walc").property("id");

let health = d3.select("#health").property("value");
let healthId = d3.select("#health").property("id");

let absences = d3.select("#absences").property("value");
let absencesId = d3.select("#absences").property("id");

let inputs = {
    schoolId: school,
    sexId: sex,
    ageId: age,
    addressId: address,
    famsizeId: famsize,
    PstatusId: Pstatus,
    MeduId: Medu,
    FeduId: Fedu,
    MjobId: Mjob,
    FjobId: Fjob,
    reasonId: reason,
    guardianId: guardian,
    traveltimeId: traveltime,
    studytimeId: studytime,
    failuresId: failures,
    schoolsupId: schoolsup,
    famsupId: famsup,
    paidId: paid,
    activitiesId: activities,
    nurseryId: nursery,
    higherId: higher,
    internetId: internet,
    romanticId: romantic,
    famrelId: famrel,
    freetimeId: freetime,
    gooutId: goout,
    DalcId: Dalc,
    WalcId: Walc,
    healthId: health,
    absencesId: absences
}

let features = {
    school: document.getElementById("school").value,
    sex: document.getElementById("sex").value,
    age: document.getElementById("age").value,
    address: document.getElementById("address").value,
    famsize: document.getElementById("famsize").value,
    Pstatus: document.getElementById("Pstatus").value,
    Medu: document.getElementById("Medu").value,
    Fedu: document.getElementById("Fedu").value,
    Mjob: document.getElementById("Mjob").value,
    Fjob: document.getElementById("Fjob").value,
    reason: document.getElementById("reason").value,
    guardian: document.getElementById("guardian").value,
    traveltime: document.getElementById("traveltime").value,
    studytime: document.getElementById("studytime").value,
    failures: document.getElementById("failures").value,
    schoolsup: document.getElementById("schoolsup").value,
    famsup: document.getElementById("famsup").value,
    paid: document.getElementById("paid").value,
    activities: document.getElementById("activities").value,
    nursery: document.getElementById("nursery").value,
    higher: document.getElementById("higher").value,
    internet: document.getElementById("internet").value,
    romantic: document.getElementById("romantic").value,
    famrel: document.getElementById("famrel").value,
    freetime: document.getElementById("freetime").value,
    goout: document.getElementById("goout").value,
    Dalc: document.getElementById("Dalc").value,
    Walc: document.getElementById("Walc").value,
    health: document.getElementById("health").value,
    absences: document.getElementById("absences").value
}



function goPython(){
    $.ajax({
        url: "something.py",
       context: document.body
      }).done(function() {
       alert('finished python script');;
      });
}
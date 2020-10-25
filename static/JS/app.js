// import the data from data.js
const tableData = data;

// Reference the HTML table using d3
var tbody = d3.select("tbody");

function buildTable(data) {
    tbody.html("");
};

data.forEach((dataRow) => {
    let row = tbody.append("tr");
});

data.forEach((dataRow) => {
    let row = tbody.append("tr");
    Object.values(dataRow).forEach((val) => {
        let cell = row.append("td");
        cell.text(val);
    }
    );
});

function buildTable(data) {
    // First, clear out any existing data
    tbody.html("");

    // Next, loop through each object in the data
    // and append a row and cells for each value in the row
    data.forEach((dataRow) => {
        // Append a row to the table body
        let row = tbody.append("tr");
  
        // Loop through each field in the dataRow and add
        // each value as a table cell (td)
        Object.values(dataRow).forEach((val) => {
            let cell = row.append("td");
            cell.text(val);
        }
        );
    });
}


var filters = {}

function handleClick() {

    let change = d3.select(this)
    let elementValue = change.property('value')
    let elementId = change.property('id')

    if (elementValue) {
        filters[elementId] = elementValue
    }
    else {
        delete filters[elementId]
    }

    filterTable()
    // // Grab the datetime value from the filter
    // let date = d3.select("#datetime").property("value");
    // let dateId = d3.select("#datetime").property("id");

    // let city = d3.select("#city").property("value");
    // let cityId = d3.select("#city").property("id");

    // let state = d3.select("#state").property("value");
    // let stateId = d3.select("#state").property("id");

    // let country = d3.select("#country").property("value");
    // let countryId = d3.select("#country").property("id");

    // let shape = d3.select("#shape").property("value");
    // let shapeId = d3.select("#shape").property("id");

    // let filters = {
    //     dateId: date,
    //     cityId: city,
    //     stateId: state,
    //     countryId: country,
    //     shapeId: shape
    // }
}

function filterTable() {

    let filteredData = tableData;

    Object.entries(filters).forEach(([key, value]) => {

        filteredData = filteredData.filter(row => row[key] === value);
    })

    buildTable(filteredData);

}




    
    // // Check to see if a date was entered and filter the
    // // data using that date.
    // if (date) {
    //     // Apply `filter` to the table data to only keep the
    //     // rows where the `datetime` value matches the filter value
    //     filteredData = filteredData.filter(row => row.datetime === date);
    // };
    // if (city) {
    //     // Apply `filter` to the table data to only keep the
    //     // rows where the `city` value matches the filter value
    //     filteredData = filteredData.filter(row => row.city === city);
    // };
    // if (state) {
    //     // Apply `filter` to the table data to only keep the
    //     // rows where the `state` value matches the filter value
    //     filteredData = filteredData.filter(row => row.state === state);
    // };
    // if (country) {
    //     // Apply `filter` to the table data to only keep the
    //     // rows where the `country` value matches the filter value
    //     filteredData = filteredData.filter(row => row.country === country);
    // };
    // if (shape) {
    //     // Apply `filter` to the table data to only keep the
    //     // rows where the `shape` value matches the filter value
    //     filteredData = filteredData.filter(row => row.shape === shape);
    // };

    // Rebuild the table using the filtered data
    // @NOTE: If no date was entered, then filteredData will
    // just be the original tableData.
    // buildTable(filteredData);
// };

// Attach an event to listen for the form button
d3.selectAll("input").on("change", handleClick);

// Build the table when the page loads
buildTable(tableData);
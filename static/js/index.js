function makesvg(percentage,thersold, inner_text=""){

    var abs_percentage = Math.abs(percentage).toString();
    var percentage_str = percentage.toString();
    var classes = ""

    if(percentage >= thersold*100){
      classes = "danger-stroke circle-chart__circle--negative";
    } else if(percentage > (thersold*100)/2 && percentage < thersold*100){
      classes = "warning-stroke";
    } else{
      classes = "success-stroke";
    }
    // viewbox="0 0 33.83098862 33.83098862"
   var svg = '<svg class="circle-chart" viewbox="0 0 33.83098862 33.83098862" xmlns="http://www.w3.org/2000/svg">'
       + '<circle class="circle-chart__background" cx="16.9" cy="16.9" r="15.9" />'
       + '<circle class="circle-chart__circle '+classes+'"'
       + 'stroke-dasharray="'+ abs_percentage+',100"    cx="16.9" cy="16.9" r="15.9" />'
       + '<g class="circle-chart__info">'
       + '   <text class="circle-chart__percent" x="17.9" y="15.5">'+percentage_str+'%</text>';

    if(inner_text){
      svg += '<text class="circle-chart__subline" x="16.91549431" y="22">'+inner_text+'</text>'
    }

    svg += ' </g></svg>';

    return svg
  }

  (function( $ ) {

      $.fn.circlechart = function() {
          this.each(function() {
              var percentage = $(this).data("percentage");
              var thersold = $(this).data('thersold');
              var inner_text = $(this).text();
              $(this).html(makesvg(percentage,thersold,inner_text));
          });
          return this;
      };

  }( jQuery ));
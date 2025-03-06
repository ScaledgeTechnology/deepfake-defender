$(document).ready(function () {
  // ------------------ Navbar ------------------
  // Toggle user dropdown
  $("#user-menu-button").click(function (event) {
    $("#user-dropdown").toggleClass("hidden");
  });

  // Toggle main menu
  $("#main-menu-button").click(function () {
    $("#navbar-user").toggleClass("hidden");
  });

  // Close dropdowns if clicked outside
  $(document).click(function (event) {
    if (!$(event.target).closest("#user-menu-button, #user-dropdown").length) {
      $("#user-dropdown").addClass("hidden");
    }
    if (!$(event.target).closest("#main-menu-button, #navbar-user").length) {
      $("#navbar-user").addClass("hidden");
    }
  });

  // ---------------- Home page (index.html) -----------------
  // Explore button scroll (to down)
  $("#exploreBtn").click(function () {
    $("html, body").animate(
      {
        scrollTop: $("#secondSection").offset().top,
      },
      1000
    ); // Smooth scroll to second section
  });

  // Scroll animation when comming to 2nd section
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          // Add animation classes when the div enters the viewport
          $(entry.target)
            .find(".div_head")
            .addClass("animate-fade-in-slide-up");
          $(entry.target).find(".para_div").addClass("animate-text-glow");
        } else {
          // Remove animation classes when the div exits the viewport
          $(entry.target)
            .find(".div_head")
            .removeClass("animate-fade-in-slide-up");
          $(entry.target).find(".para_div").removeClass("animate-text-glow");
        }
      });
    },
    {
      threshold: 0.2, // Trigger when 20% of the element is visible
    }
  );

  // Observe each scroll-animate div
  $(".scroll-animate").each(function () {
    observer.observe(this);
  });

  // -------------- upload video page (upload_video.html) -----------------

});

// Open and close sidebar
var open = false;
function openNav() {
  if (!open) {
    
  document.getElementById("sidebar").style.width = "250px";
  document.getElementById("main-content").style.transform='translateX(250px)';
  
  open = true;
  }
  else {
    closeNav();
  }
}

function closeNav() {
  document.getElementById("sidebar").style.width = "0";
  document.getElementById("main-content").style.transform = 'translateX(0)';
  open = false;
}
 
function previewImage(event) {
  const fileInput = event.target;
  const reader = new FileReader();

  reader.onload = function () {
      const preview = document.getElementById('image-preview');
      preview.src = reader.result;
      preview.style.display = 'block'; // عرض الصورة
  };

  if (fileInput.files[0]) {
      reader.readAsDataURL(fileInput.files[0]); // قراءة الملف
  }
}

const carousel = document.getElementById("carousel");
const items = document.querySelectorAll(".ad-item");
const totalItems = items.length;
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
let currentIndex = 0;

// Function to update carousel position
function updateCarousel() {
    const offset = -currentIndex * 100;
    carousel.style.transform = `translateX(${offset}%)`;
}

// Event listeners for navigation buttons
prevBtn.addEventListener("click", () => {
    currentIndex = (currentIndex - 1 + totalItems) % totalItems;
    updateCarousel();
});

nextBtn.addEventListener("click", () => {
    currentIndex = (currentIndex + 1) % totalItems;
    updateCarousel();
});

document.addEventListener("DOMContentLoaded", function () {
  const items = document.querySelectorAll(".ad-item");
  let currentIndex = 0;

  function showItem(index) {
      items.forEach((item, i) => {
          item.classList.remove("active");
          if (i === index) {
              item.classList.add("active");
          }
      });
  }

  document.getElementById("prevBtn").addEventListener("click", function () {
      currentIndex = (currentIndex - 1 + items.length) % items.length;
      showItem(currentIndex);
  });

  document.getElementById("nextBtn").addEventListener("click", function () {
      currentIndex = (currentIndex + 1) % items.length;
      showItem(currentIndex);
  });

  // عرض أول عنصر عند تحميل الصفحة
  showItem(currentIndex);
});

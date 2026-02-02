function sendToNative(url) {
  return browser.runtime.sendNativeMessage("minrss_native_host", { url });
}

function notify(title, message) {
  return browser.notifications.create({
    type: "basic",
    title,
    message,
  });
}

browser.menus.create({
  id: "minrss-add",
  title: "Add feed to minrss",
  contexts: ["link"]
});

browser.menus.onClicked.addListener((info) => {
  if (!info.linkUrl) return;
  sendToNative(info.linkUrl)
    .then((res) => {
      if (res && res.ok) {
        notify("minrss", "Feed added");
      } else {
        const msg = res && (res.error || res.stderr) ? (res.error || res.stderr) : "Add failed";
        notify("minrss", msg);
      }
    })
    .catch((err) => {
      console.error("minrss native message failed", err);
      notify("minrss", "Add failed");
    });
});
